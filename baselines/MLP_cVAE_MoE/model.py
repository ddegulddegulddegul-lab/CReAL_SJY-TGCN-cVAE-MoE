import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# To import utils regardless of running directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))

class MLPEncoder(nn.Module):
    """
    Naive baseline that flattens all spatial relationships and uses standard Linear layers.
    Demonstrates the baseline performance without Spatio-Temporal Graph Convolutions.
    """
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_joints=15, num_rays=540):
        super(MLPEncoder, self).__init__()
        self.num_joints = num_joints
        self.num_rays = num_rays
        
        flat_dim = (num_joints * node_in_dim) + (num_rays * ray_in_dim)
        encoded_dim = hidden_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, encoded_dim * 2),
            nn.ReLU(),
            nn.Linear(encoded_dim * 2, encoded_dim),
            nn.ReLU()
        )
        
        # Ray specific encoder to provide spatial MoE condition matching TGCN dim
        self.ray_mlp = nn.Sequential(
            nn.Linear(ray_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim),
            nn.ReLU()
        )
        
        self.mu_net = nn.Linear(encoded_dim, z_dim)
        self.logvar_net = nn.Linear(encoded_dim, z_dim)
        
    def forward(self, node_feats, ray_feats):
        B, T, _, _ = node_feats.shape
        
        # [Fallback Safety Check] Ensure ray_feats has the 4D (XYZ + Distance) dimension
        if ray_feats.shape[-1] == 3:
            ray_dist = torch.norm(ray_feats, p=2, dim=-1, keepdim=True)
            ray_feats = torch.cat([ray_feats, ray_dist], dim=-1)
            
        # 1. Spatially Flatten the nodes and rays for standard MLP
        N_flat = node_feats.view(B, T, -1)
        R_flat = ray_feats.view(B, T, -1)
        
        x = torch.cat([N_flat, R_flat], dim=-1) # (B, T, flat_dim)
        
        # 2. Extract Global Scene Feature
        scene_feat = self.mlp(x) # (B, T, encoded_dim)
        
        mu = self.mu_net(scene_feat)
        logvar = self.logvar_net(scene_feat)
        
        # 3. Extract independent Ray encodings for Decoder condition
        encoded_rays = self.ray_mlp(ray_feats) # (B, T, 540, encoded_dim)
        
        return mu, logvar, encoded_rays

class MoEDecoder(nn.Module):
    # Identical to the proposed model for fair Encoder comparison ablation
    def __init__(self, in_channels, out_dim=6, num_experts=3):
        super(MoEDecoder, self).__init__()
        self.num_experts = num_experts
        hidden_dim = in_channels // 2
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            ) for _ in range(num_experts)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, z_c):
        gate_weights = self.gate(z_c) 
        expert_outputs = torch.stack([expert(z_c) for expert in self.experts], dim=-1) 
        out = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1) 
        return torch.relu(out) 

class IIWMLP_cVAE(nn.Module):
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_body_parts=6, num_experts=3):
        super(IIWMLP_cVAE, self).__init__()
        self.encoder = MLPEncoder(node_in_dim, ray_in_dim, hidden_dim, z_dim)
        
        decoder_in_dim = z_dim + (hidden_dim * 2)
        self.decoder = MoEDecoder(in_channels=decoder_in_dim, out_dim=num_body_parts, num_experts=num_experts)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, node_feats, ray_feats):
        B, T, _, _ = node_feats.shape
        num_rays = ray_feats.shape[2]
        
        mu, logvar, encoded_rays = self.encoder(node_feats, ray_feats)
        
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu 
            
        z_expanded = z.unsqueeze(2).expand(B, T, num_rays, -1)
        z_c = torch.cat([z_expanded, encoded_rays], dim=-1) 
        
        pred_iiw = self.decoder(z_c) 
        pred_iiw = pred_iiw.permute(0, 1, 3, 2).contiguous()
        
        return pred_iiw, mu, logvar
