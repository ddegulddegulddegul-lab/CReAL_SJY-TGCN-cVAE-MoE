import time
import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import viser

# Bind to shared utils and local model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from dataset import get_dataloader
from model import IIWTGCN_cVAE

def get_all_data_files(base_dir):
    """Recursively scans the provided directory for .npz files."""
    npz_files = []
    if os.path.exists(base_dir):
        for root, _, files in os.walk(base_dir):
            for f in sorted(files):
                if f.endswith('.npz'):
                    npz_files.append(os.path.join(root, f))
    return npz_files

def pre_load_all_data(weights_path, data_files, device):
    """Inferences all sequences into CPU memory to prevent rendering lag in Viser GUI."""
    if not data_files:
        print("Warning: No .npz files found in the specified data directory!")
        return {}
        
    model = IIWTGCN_cVAE().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}. Showing untrained model outputs.")
        
    model.eval()
    loaded_data = {}
    print(f"Pre-loading {len(data_files)} datasets. Generating IIW Physics...")
    
    for path in data_files:
        # parentDir/filename key grouping
        parent_dir = os.path.basename(os.path.dirname(path))
        filename = f"{parent_dir}/{os.path.basename(path)}"
        
        test_data = np.load(path)['clips']
        
        # We process the entire sequence as one batch for visualization
        _, dataloader = get_dataloader(
            test_data, seq_len=test_data.shape[0], stride=1, batch_size=1, shuffle=False, num_workers=0, split_mode='all'
        )
        batch = next(iter(dataloader))
        
        node_feats = batch['node_features'].to(device)
        ray_feats = batch['ray_features'].to(device)
        
        with torch.no_grad():
            pred_iiw, _, _ = model(node_feats, ray_feats)
            
        n_cpu = node_feats.cpu().numpy()
        loaded_data[filename] = {
            "node_feats": n_cpu,
            "ray_feats": ray_feats.cpu().numpy(),
            "pred_iiw": pred_iiw.cpu().numpy(),
            "T": n_cpu.shape[1]
        }
        
    print("Pre-loading generation complete!")
    return loaded_data

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Preparing 3D Physics Viser Server...")
    
    # 1. Pre-load logic
    data_files = get_all_data_files(args.data_dir)
    dataset_dict = pre_load_all_data(args.weights_path, data_files, device)
    file_options = list(dataset_dict.keys())
    
    if not file_options:
        print("Error: No datasets available. Check --data_dir parameter.")
        return
        
    current_key = file_options[0]
    node_feats = dataset_dict[current_key]["node_feats"]
    ray_feats = dataset_dict[current_key]["ray_feats"]
    pred_iiw = dataset_dict[current_key]["pred_iiw"]
    T = dataset_dict[current_key]["T"]
    
    # 2. Setup Viser
    server = viser.ViserServer(port=args.port)
    print("\n" + "="*50)
    print(f"✅ Viser Vector Graphics Server running at: http://localhost:{args.port}")
    try:
        share_url = server.request_share_url()
        print(f"🌍 STABLE PUBLIC link: {share_url}")
    except:
        pass
    print("="*50 + "\n")
    
    # 3. Add base plane
    server.scene.add_grid("/floor_grid", width=10.0, height=10.0, cell_size=0.5, plane="xy")
    
    body_parts = ["All Parts", "base/pelvis", "spine", "right hand", "left hand", "right foot", "left foot"]
    
    file_dropdown = server.gui.add_dropdown("Dataset Sequence", options=file_options, initial_value=file_options[0])
    with server.gui.add_folder("Playback Controls"):
        play_button = server.gui.add_button("Play / Pause")
        time_slider = server.gui.add_slider("Time Frame (T)", min=0, max=T-1, step=1, initial_value=0)
        part_dropdown = server.gui.add_dropdown("Body Part Heatmap", options=body_parts, initial_value="All Parts")
        
    is_playing = False
    
    @file_dropdown.on_update
    def _(_):
        nonlocal node_feats, ray_feats, pred_iiw, T, is_playing
        is_playing = False
        k = file_dropdown.value
        node_feats = dataset_dict[k]["node_feats"]
        ray_feats = dataset_dict[k]["ray_feats"]
        pred_iiw = dataset_dict[k]["pred_iiw"]
        T = dataset_dict[k]["T"]
        time_slider.max = T - 1
        time_slider.value = 0
        update_scene()
        
    @play_button.on_click
    def _(_):
        nonlocal is_playing
        is_playing = not is_playing
        
    # Bone connections for 15 joints
    bone_edges = np.array([
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
        [7, 8], [7, 11], [7, 12], [8, 9], [9, 10], [12, 13], [13, 14]
    ])
    
    def transform_coords(pts):
        pts_viser = np.zeros_like(pts)
        pts_viser[:, 0] = pts[:, 0]
        pts_viser[:, 1] = pts[:, 2]
        pts_viser[:, 2] = pts[:, 1]
        return pts_viser
        
    def update_scene():
        t = time_slider.value
        part_idx = body_parts.index(part_dropdown.value)
        
        j_pos = transform_coords(node_feats[0, t, :, :3])
        r_pos = transform_coords(ray_feats[0, t, :, :3])
        
        server.scene.add_point_cloud(
            "/human/joints", points=j_pos, colors=np.array([0, 0, 0], dtype=np.uint8), point_size=0.03
        )
        
        for i, (u, v) in enumerate(bone_edges):
            pts = np.stack([j_pos[u], j_pos[v]])
            server.scene.add_spline_catmull_rom(f"/human/bones/edge_{i}", positions=pts, color=(0,0,0), line_width=4.0)
            
        colors_float = np.full((540, 3), [0.8, 0.8, 0.8], dtype=np.float32)
        
        if part_dropdown.value == "All Parts":
            all_weights = pred_iiw[0, t, :, :]
            max_w = np.max(all_weights, axis=0)
            dom_p = np.argmax(all_weights, axis=0) # 0 to 5
            
            for i in range(540):
                w = max_w[i]
                if w >= 0.1:
                    norm_w = max(0.0, min(1.0, (w - 0.1) / 0.9))  # 0.1~1.0 -> 0~1
                    # 0: base(Red), 1: spine(Purple), 2/3: hands(Orange), 4/5: feet(Blue)
                    c_map = ["Reds", "Purples", "Oranges", "Oranges", "Blues", "Blues"][dom_p[i]]
                    colors_float[i] = plt.get_cmap(c_map)(norm_w)[:3]
                    
            # Spine override removed at user request for equal visualization
        else:
            # 파츠별 고유 색: pelvis=Red, spine=Purple, r.hand=Orange, l.hand=Yellow, r.foot=Blue, l.foot=Cyan
            part_cmaps = ["Reds", "Purples", "Oranges", "YlOrBr", "Blues", "GnBu"]
            c_map = part_cmaps[part_idx - 1]
            w = pred_iiw[0, t, part_idx - 1, :]
            mask = w >= 0.1
            if np.any(mask):
                norm_w = np.clip((w[mask] - 0.1) / 0.9, 0.0, 1.0)  # 0.1~1.0 -> 0~1
                colors_float[mask] = plt.get_cmap(c_map)(norm_w)[:, :3]
                
        server.scene.add_point_cloud(
            "/furniture_rays", points=r_pos, colors=(colors_float * 255).astype(np.uint8), point_size=0.02
        )
        
    @time_slider.on_update
    def _(_): update_scene()
    @part_dropdown.on_update
    def _(_): update_scene()
    
    update_scene()
    while True:
        if is_playing:
            time_slider.value = (time_slider.value + 1) % T
            time.sleep(0.05)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize TGCN IIW Model")
    parser.add_argument("--data_dir", type=str, default="/home/song/reserch/iiw/data/", help="Folder containing .npz sequences")
    parser.add_argument("--weights_path", type=str, default="./weights/best.pth", help="Path to trained model weights")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    main(args)
