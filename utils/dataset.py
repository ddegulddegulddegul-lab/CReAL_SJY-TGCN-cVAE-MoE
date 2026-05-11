import torch
from torch.utils.data import Dataset, DataLoader
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
            'gt_iiw': gt_iiw
        }

def get_dataloader(data_array, seq_len=30, stride=1, batch_size=32, shuffle=True, num_workers=8, split_mode='all'):
    """
    Returns the initialized dataset and its corresponding DataLoader for the specified split.
    """
    dataset = IIWDataset(data_array, seq_len=seq_len, stride=stride, split_mode=split_mode)
    
    # If the dataset is too small for a batch, prevent crashes
    if len(dataset) == 0:
        return dataset, None
        
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    return dataset, dataloader
