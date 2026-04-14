import time
import viser
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import IIWTGCN_cVAE
from dataset import get_dataloader

def get_all_data_files():
    """/home/song/reserch/iiw/data 하위 3개 폴더에서 첫 번째 .npz 파일만 가져옵니다. (총 3개)"""
    base_dir = '/home/song/reserch/iiw/data'
    sub_dirs = ['Source_Extraction', 'hc_hd_Extraction', 'lc_ld_b_Extraction']
    
    npz_files = []
    for sub in sub_dirs:
        target_dir = os.path.join(base_dir, sub)
        if os.path.isdir(target_dir):
            found_one = False
            for root, _, files in os.walk(target_dir):
                if found_one: break
                for f in sorted(files):
                    if f.endswith('.npz'):
                        npz_files.append(os.path.join(root, f))
                        found_one = True
                        break # 하나 찾으면 다음 최상위 폴더로 넘어가기 위함
    return npz_files

def pre_load_all_data():
    """
    지정된 3개의 데이터를 미리 추론하여 메모리(dict)에 올려둡니다. (GUI 렌더링 멈춤 방지)
    """
    weights_path = '/home/song/reserch/iiw/tgcn_cvae_weights.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_files = get_all_data_files()
    if not data_files:
        print("Warning: No data files found!")
        return {}
        
    model = IIWTGCN_cVAE().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    loaded_data = {}
    print(f"Pre-loading {len(data_files)} datasets into memory...")
    
    for path in data_files:
        # 상위 폴더 이름까지 포함시켜서 key가 중복되지 않도록 만듭니다. (예: hc_hd_Extraction/hc_hd_bl_1)
        parent_dir = os.path.basename(os.path.dirname(path))
        filename = f"{parent_dir}/{os.path.basename(path)}"
        
        print(f" > Inferencing: {filename}")
        test_data = np.load(path)['clips']
        
        _, dataloader = get_dataloader(
            test_data, seq_len=test_data.shape[0], stride=1, batch_size=1, shuffle=False, num_workers=0
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
        
    print("Pre-loading complete!")
    return loaded_data

def main():
    print("Preparing 3D Viser Server...")
    
    # 1. 시퀀스 텐서 리스트 준비 (Pre-load)
    dataset_dict = pre_load_all_data()
    file_options = list(dataset_dict.keys())
    
    if not file_options:
        print("Error: No datasets available.")
        return
        
    # 초기 데이터 세팅
    current_key = file_options[0]
    node_feats = dataset_dict[current_key]["node_feats"]
    ray_feats = dataset_dict[current_key]["ray_feats"]
    pred_iiw = dataset_dict[current_key]["pred_iiw"]
    T = dataset_dict[current_key]["T"]
    
    # 2. Viser 서버 시작 (share=True 를 통해 안정적인 외부 접속 터널링 URL 생성)
    server = viser.ViserServer(port=8080)
    
    print("\n" + "="*50)
    print("✅ Viser Server is running!")
    print("👉 Open your LOCAL browser at: http://localhost:8080")
    try:
        share_url = server.request_share_url()
        print(f"🌍 Or use this STABLE PUBLIC link: {share_url}")
    except Exception as e:
        print(f"⚠️ Could not generate public share link: {e}")
    print("="*50 + "\n")
    
    # [Point 3] 바닥면(XZ 평면) 그리드 추가
    # 캐릭터가 서 있는 공간 크기를 가늠할 수 있도록 10x10 크기의 그리드를 원점에 생성합니다.
    server.scene.add_grid(
        "/floor_grid", 
        width=10.0, 
        height=10.0,
        cell_size=0.5,
        plane="xy" # XZ 평면이 바닥면 역할을 함
    )
    
    body_parts = ["All Parts (Aggregated)", "base/pelvis", "spine", "right hand", "left hand", "right foot", "left foot"]
    
    # 3. GUI 컨트롤러 추가
    file_dropdown = server.gui.add_dropdown("Dataset File", options=file_options, initial_value=file_options[0])
    
    with server.gui.add_folder("Playback Controls"):
        play_button = server.gui.add_button("Play / Pause")
        time_slider = server.gui.add_slider(
            "Time Frame (T)", min=0, max=T-1, step=1, initial_value=0
        )
        part_dropdown = server.gui.add_dropdown(
            "Body Part", options=body_parts, initial_value="All Parts (Aggregated)"
        )
        
    @file_dropdown.on_update
    def _(_):
        nonlocal node_feats, ray_feats, pred_iiw, T, is_playing
        is_playing = False
        
        selected_file = file_dropdown.value
        node_feats = dataset_dict[selected_file]["node_feats"]
        ray_feats = dataset_dict[selected_file]["ray_feats"]
        pred_iiw = dataset_dict[selected_file]["pred_iiw"]
        T = dataset_dict[selected_file]["T"]
        
        time_slider.max = T - 1
        time_slider.value = 0
        update_scene()
        
    # [Point 1] Colormap 준비 (uint8 변환을 위해 matplotlib get_cmap 이용)
    colormap = plt.get_cmap("coolwarm")
    
    # [Point 2] 15개 관절을 이어줄 뼈대(Bone) 연결 인덱스 정의 (휴리스틱 트리 구조)
    bone_edges = np.array([
        [0, 1],   # Pelvis -> Spine
        [1, 2],   # Spine -> Chest/Neck
        [2, 3],   # Chest -> R_Shoulder
        [0, 4],   # Pelvis -> L_Shoulder
        [4, 5],   # R_Elbow -> R_Hand
        [5, 6],   # R_Hand -> R_Finger
        [0, 7],
        [7, 8],   # L_Elbow -> L_Hand
        [7, 11], # R_Knee -> R_Foot
        [7, 12],  # Pelvis -> L_Hip
        [8, 9],   # R_Hand -> R_Finger
        [9, 10],  # R_Hip -> R_Knee
        [12, 13], # L_Hip -> L_Knee
        [13, 14]  # L_Knee -> L_Foot
    ])
    
    # 재생 상태 변수
    is_playing = False
    
    @play_button.on_click
    def _(_):
        nonlocal is_playing
        is_playing = not is_playing
    
    # 4. 실시간 Scene 업데이트 함수
    def update_scene():
        t = time_slider.value
        part_idx = body_parts.index(part_dropdown.value)
        
        # [A] Human Pose 렌더링 (15 관절, XYZ 좌표 추출)
        joints_pos = node_feats[0, t, :, :3]
        
        # [B] Furniture Rays 렌더링 (540 포인트, 히트맵 컬러 적용)
        rays_pos = ray_feats[0, t, :, :] # (540, 3)
        weights = pred_iiw[0, t, part_idx, :] # (540,) 가중치 (0.0 ~ 1.0)
        
        # --- Unity(Y-Up) to Viser(Z-Up) 좌표 변환 ---
        def transform_coords(pts):
            pts_viser = np.zeros_like(pts)
            pts_viser[:, 0] = pts[:, 0]      # X는 그대로
            pts_viser[:, 1] = pts[:, 2]      # Unity의 Z축이 Viser의 Y축(깊이)으로
            pts_viser[:, 2] = pts[:, 1]     # Unity의 Y축(안티/중력)이 Viser의 Z축(높이) 역방향으로
            return pts_viser
            
        joints_pos = transform_coords(joints_pos)
        rays_pos = transform_coords(rays_pos)
        
        # 관절 인덱스 레이블 추가 (WebGL 오버로드 화이트 스크린 방지를 위해 제거)
        # 1초에 20번씩 15개의 텍스트 위치를 리렌더링하면 Viser 프론트엔드 React가 죽는 현상 해결
        
        # 관절 포인트 렌더링 (검은색)
        server.scene.add_point_cloud(
            "/human/joints",
            points=joints_pos,
            colors=np.array([0, 0, 0], dtype=np.uint8), # 검은색
            point_size=0.02
        )
        
        # [Point 2] 뼈대(Bones/Edges) 그리기 (검은색)
        for i, (u, v) in enumerate(bone_edges):
            pts = np.stack([joints_pos[u], joints_pos[v]])
            server.scene.add_spline_catmull_rom(
                f"/human/bones/edge_{i}",
                positions=pts,
                color=(0, 0, 0), # 검은색
                line_width=3.0,
                tension=0.0
            )
        
        # --- Colorize Rays ---
        colors_float = np.full((540, 3), [0.6, 0.6, 0.6], dtype=np.float32) # 연한 회색 초기화
        
        selected_part = part_dropdown.value
        
        if selected_part == "All Parts (Aggregated)":
            # 6개 파트의 모든 IIW
            all_weights = pred_iiw[0, t, :, :] # (6, 540)
            
            max_weights = np.max(all_weights, axis=0) # (540,)
            dominant_parts = np.argmax(all_weights, axis=0) # (540,) - 0부터 5까지의 파트 인덱스
            
            # --- 1차 글로벌 렌더링 (가장 높은 가중치를 가진 부위 베이스) ---
            for i in range(540):
                w = max_weights[i]
                p_idx = dominant_parts[i]
                
                if w >= 0.6:
                    norm_w = (w - 0.6) / (1.0 - 0.6)
                    
                    if p_idx == 0:     cmap = plt.get_cmap("Reds")     # Pelvis = 붉은색
                    elif p_idx == 1:   cmap = plt.get_cmap("Purples")  # Spine = 보라색
                    elif p_idx in (2, 3): cmap = plt.get_cmap("Wistia")  # Hands = 노란색
                    else:                 cmap = plt.get_cmap("Blues")    # Feet = 파란색
                        
                    colors_float[i] = cmap(norm_w)[:3]
                    
            # --- 2차 Spine(허리) 전용 강제 덧칠 (Overlay) ---
            # 허리(Spine)의 고유 IIW 값이 엉덩이(Pelvis)의 IIW 값보다 낮아서 지워지지만,
            # 엉덩이가 확실하게 닿은 부위(0.8 이상의 딥레드 컬러)는 엉덩이의 영역으로 인정하고 보존합니다.
            spine_weights = all_weights[1, :]
            pelvis_weights = all_weights[0, :]
            for i in range(540):
                sw = spine_weights[i]
                pw = pelvis_weights[i]
                
                # 허리 값이 0.6 이상이면서, 엉덩이 값이 0.8 미만(딥레드가 아님)일 때만 덧칠함
                if sw >= 0.6 and pw < 0.8:
                    # 허리 전용 정규화
                    norm_sw = (sw - 0.6) / (1.0 - 0.6)
                    # 기존 픽셀 색상을 덮어쓰고 Spine 퍼플 그라데이션 적용
                    colors_float[i] = plt.get_cmap("Purples")(norm_sw)[:3]
                    
                    
        else:
            # 기존 단일 파트 렌더링
            # "All Parts" 가 0인덱스에 있으므로 기존 부위 인덱스를 구하기 위해 1을 빼야함
            part_idx = body_parts.index(selected_part) - 1 
            weights = pred_iiw[0, t, part_idx, :] # (540,)
            
            active_mask = weights >= 0.6
            if np.any(active_mask):
                normalized_weights = (weights[active_mask] - 0.6) / (1.0 - 0.6)
                colors_float[active_mask] = plt.get_cmap("Reds")(normalized_weights)[:, :3]
                
        colors_uint8 = (colors_float * 255.0).astype(np.uint8)
        
        server.scene.add_point_cloud(
            "/furniture_rays",
            points=rays_pos,
            colors=colors_uint8,
            point_size=0.02 # 레이 점들의 크기를 작게 축소 (0.05 -> 0.02)
        )
        
    # 슬라이더 값 변경(업데이트) 이벤트 바인딩
    @time_slider.on_update
    def _(_):
        update_scene()
        
    # 신체 부위 선택 변경(업데이트) 이벤트 바인딩
    @part_dropdown.on_update
    def _(_):
        update_scene()
        
    # 서버 실행 시 최초 1회 렌더링 트리거
    update_scene()
    
    # 메인 루프 (서버 유지 밎 Auto Playback 처리)
    while True:
        if is_playing:
            # 프레임을 1씩 증가시키고, 끝에 도달하면 0으로 래핑
            time_slider.value = (time_slider.value + 1) % T
            # 초당 20프레임 속도 조절
            time.sleep(0.05)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
