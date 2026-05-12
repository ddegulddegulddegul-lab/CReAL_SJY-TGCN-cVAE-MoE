import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

class IIWDataset(Dataset):
    def __init__(self, data_array, seq_len=30, stride=1, split_mode='all'):
        """
        Args:
            data_array (np.ndarray or torch.Tensor): The raw 1D array time-series data.
            seq_len (int): Length of the time sequence (T frames).
            stride (int): Sliding window stride to extract sequences.
            split_mode (str): Data splitting mode ['train', 'val', 'test', 'all'].
                              Uses chronological 80/10/10 frame splits.
        """
        self.seq_len = seq_len
        self.stride = stride
        
        # Perform logical slicing to create subsets before processing
        total_raw_frames = data_array.shape[0]
        
        if split_mode == 'train':
            start_idx = 0
            end_idx = int(total_raw_frames * 0.8)
        elif split_mode == 'val':
            start_idx = int(total_raw_frames * 0.8)
            end_idx = int(total_raw_frames * 0.9)
        elif split_mode == 'test':
            start_idx = int(total_raw_frames * 0.9)
            end_idx = total_raw_frames
        elif split_mode == 'all':
            start_idx = 0
            end_idx = total_raw_frames
        else:
            raise ValueError("split_mode must be one of: 'train', 'val', 'test', 'all'")
            
        # For large memmap arrays, just hold the sliced reference
        self.data = data_array[start_idx:end_idx]
        self.total_frames = self.data.shape[0]
        
        # Calculate the number of valid sequences we can extract
        if self.total_frames < self.seq_len:
            self.num_sequences = 0
            print(f"Warning: {split_mode} split frames ({self.total_frames}) is less than seq_len ({self.seq_len})")
        else:
            self.num_sequences = (self.total_frames - self.seq_len) // self.stride + 1
            
    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        
        # Extract the sequence from memmap as a numpy array, then convert to tensor
        seq_data_np = self.data[start_idx:end_idx]
        seq_data = torch.tensor(seq_data_np, dtype=torch.float32)
        
        # 1. Key Joint Pose (Node Feature): [0:135] -> (T, 15, 9)
        node_features = seq_data[:, 0:135].view(self.seq_len, 15, 9)
        
        # 2. Ray Hit Points (Ray Feature): [204:1824] -> (T, 540, 3)
        ray_features = seq_data[:, 204:1824].view(self.seq_len, 540, 3)
        
        # --- Advanced Feature Engineering: [Point 1] L2 Norm Distance Concatenation ---
        # Calculate Euclidean distance from origin (0,0,0) for each ray target.
        # If the ray hit nothing, the position is [10,10,10], which creates a huge distance 
        # (approx 17.32) that acts as a strong natural spatial filter for the TGCN.
        ray_distances = torch.norm(ray_features, p=2, dim=-1, keepdim=True) # (T, 540, 1)
        
        # 4D Ray Feature: XYZ + Distance -> (T, 540, 4)
        ray_features = torch.cat([ray_features, ray_distances], dim=-1)
        
        # 3. Ground Truth IIW: [7224:10464] -> (T, 6, 540)
        gt_iiw = seq_data[:, 7224:10464].view(self.seq_len, 6, 540)
        
        return {
            'node_features': node_features,
            'ray_features': ray_features,   # Now 4-dimensional!
            'gt_iiw': gt_iiw,
            'part_active': (gt_iiw > 0.0).any(dim=-1).float()
        }

    def sequence_part_flags(self, threshold=0.0, chunk_size=4096):
        """
        Returns a (num_sequences, 6) boolean array indicating which body parts
        have any active IIW inside each temporal training window.
        """
        if self.num_sequences == 0:
            return np.zeros((0, 6), dtype=bool)

        frame_flags = np.zeros((self.total_frames, 6), dtype=np.bool_)
        for start in range(0, self.total_frames, chunk_size):
            end = min(self.total_frames, start + chunk_size)
            gt = self.data[start:end, 7224:10464].reshape(end - start, 6, 540)
            frame_flags[start:end] = (gt > threshold).any(axis=2)

        cumsum = np.concatenate(
            [np.zeros((1, 6), dtype=np.int32), np.cumsum(frame_flags, axis=0, dtype=np.int32)],
            axis=0
        )
        starts = np.arange(self.num_sequences) * self.stride
        ends = starts + self.seq_len
        counts = cumsum[ends] - cumsum[starts]
        return counts > 0


def build_contact_balanced_weights(
    dataset,
    threshold=0.0,
    part_weights=(1.0, 3.0, 10.0, 10.0, 1.0, 1.0),
    base_weight=1.0,
):
    """
    Builds sequence-level sampling weights so rare hand/spine contact windows
    are not drowned out by pelvis/foot and all-zero windows.
    """
    flags = dataset.sequence_part_flags(threshold=threshold)
    if len(flags) == 0:
        return np.array([], dtype=np.float64)

    part_weights = np.asarray(part_weights, dtype=np.float64)
    weights = np.full(flags.shape[0], base_weight, dtype=np.float64)
    weights += flags.astype(np.float64) @ part_weights
    return weights


def get_dataloader(
    data_array,
    seq_len=30,
    stride=1,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    split_mode='all',
    balance_contacts=False,
    sampler_part_weights=(1.0, 3.0, 10.0, 10.0, 1.0, 1.0),
    sampler_threshold=0.0,
):
    """
    Returns the initialized dataset and its corresponding DataLoader for the specified split.
    """
    dataset = IIWDataset(data_array, seq_len=seq_len, stride=stride, split_mode=split_mode)
    
    # If the dataset is too small for a batch, prevent crashes
    if len(dataset) == 0:
        return dataset, None
        
    sampler = None
    if balance_contacts:
        weights = build_contact_balanced_weights(
            dataset,
            threshold=sampler_threshold,
            part_weights=sampler_part_weights,
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    return dataset, dataloader
