import torch
import torch.nn as nn

class AdaptiveGraphAdjacency(nn.Module):
    """
    Constructs an adaptive Bipartite Block Adjacency Matrix for Heterogeneous Nodes:
    15 Joint Nodes + 540 Ray Nodes = Total 555 Nodes.
    
    The strategy is:
    1. Base Topology (A_base) -> Fixed Structural Edges:
       - A_{human-human}: Fully connected for 15 nodes (Structural rigidity).
       - A_{ray-ray}: Local sliding window connections (Spherical neighbor topology).
       
    2. Adaptive Bipartite Topology (A_{human-ray}):
       - [Key Novelty] Dynamically initialized per-frame using the INVERSE EUCLIDEAN DISTANCE.
       - A_dyn = 1 / (Distance(XYZ_human, XYZ_ray) + epsilon)
       - The learnable matrix `PA` acts as an attention offset over this physical baseline.
    """
    def __init__(self, num_joints=15, num_rays=540, ray_vertical=140, ray_horizontal=400):
        super(AdaptiveGraphAdjacency, self).__init__()
        self.num_joints = num_joints
        self.num_rays = num_rays
        self.total_nodes = num_joints + num_rays
        
        assert num_rays == (ray_vertical + ray_horizontal), "Ray split sum must be equal to num_rays"
        
        # --- 1. Base Topology Construction ---
        A_base = torch.zeros(self.total_nodes, self.total_nodes)
        
        # Block 1: Joint to Joint [0:15, 0:15]
        # In a Bipartite setup, we keep the internal structure solid.
        A_base[:num_joints, :num_joints] = 1.0 
        
        # Block 2: Ray to Ray [15:555, 15:555]
        k_neighbor = 2
        
        v_start = num_joints
        v_end = num_joints + ray_vertical
        for i in range(v_start, v_end):
            for j in range(max(v_start, i-k_neighbor), min(v_end, i+k_neighbor+1)):
                A_base[i, j] = 1.0
                
        h_start = v_end
        h_end = self.total_nodes
        for i in range(h_start, h_end):
            for j in range(max(h_start, i-k_neighbor), min(h_end, i+k_neighbor+1)):
                A_base[i, j] = 1.0
        
        # Add Self-loops
        A_base.fill_diagonal_(1.0)
        A_base = torch.clamp(A_base + A_base.T, max=1.0)
        
        # Normalize
        D = torch.sum(A_base, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
        A_norm = torch.matmul(torch.matmul(D_mat_inv_sqrt, A_base), D_mat_inv_sqrt)
        
        self.register_buffer('A_base', A_norm)
        
        # --- 2. Learnable Adjacency (PA) ---
        self.PA = nn.Parameter(torch.zeros(self.total_nodes, self.total_nodes))
        # Initialize as 0 so the initial flow exactly matches the Inverse Distance math
        nn.init.zeros_(self.PA)

    def forward(self, node_xyz, ray_xyz):
        """
        Dynamically calculate adjacency for the current batch.
        Args:
            node_xyz: (B, T, 15, 3) - Positions of human joints
            ray_xyz:  (B, T, 540, 3) - Hit positions of furniture rays
        Returns:
            A_final: (B*T, 555, 555)
        """
        B, T, V_n, _ = node_xyz.shape
        _, _, V_r, _ = ray_xyz.shape
        
        # Flatten time into batch for sequence graph generation
        node_xyz_flat = node_xyz.view(B*T, V_n, 3)
        ray_xyz_flat = ray_xyz.view(B*T, V_r, 3)
        
        # 1. Calculate Pairwise Euclidean Distance
        # (B*T, 15, 3) & (B*T, 540, 3) -> cdist calculates distance matrix of shape (B*T, 15, 540)
        dist = torch.cdist(node_xyz_flat, ray_xyz_flat, p=2)
        
        # 2. Convert to Inverse Distance (Closer = Higher Graph Edge Weight)
        eps = 1e-4 # Prevent division by zero
        inv_dist = 1.0 / (dist + eps)
        
        # Optional: Normalize inverse distances per row (softmax or max division) to prevent exploding values
        inv_dist = inv_dist / (torch.max(inv_dist, dim=-1, keepdim=True)[0] + eps)
        
        # 3. Create Dynamic Bipartite Connections (A_dyn)
        A_dyn = torch.zeros(B*T, self.total_nodes, self.total_nodes, device=node_xyz.device)
        
        # Apply Inverse Distance strictly to Human-Ray and Ray-Human blocks
        A_dyn[:, :self.num_joints, self.num_joints:] = inv_dist
        A_dyn[:, self.num_joints:, :self.num_joints] = inv_dist.transpose(1, 2)
        
        # 4. Create a Bipartite Mask to restrict where PA can learn
        mask = torch.zeros_like(self.PA)
        mask[:self.num_joints, self.num_joints:] = 1.0  # Human -> Ray
        mask[self.num_joints:, :self.num_joints] = 1.0  # Ray -> Human

        # Apply mask to PA before adding
        PA_masked = self.PA * mask

        # 5. Combine Base + Dynamic Inverse Dist + Masked Learnable Attention
        A_combined = self.A_base.unsqueeze(0) + A_dyn + PA_masked.unsqueeze(0)
        
        # 6. Non-negative enforcement and Gradient Clipping
        A_final = torch.relu(A_combined)
        A_final = torch.clamp(A_final, min=0.0, max=2.0)
        
        return A_final
