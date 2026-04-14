import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# To import utils regardless of running directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from graph import AdaptiveGraphAdjacency

class SpatialGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialGCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        B, C, T, V = x.size()
        
        # 1. Linear Projection
        x_mapped = x.permute(0, 2, 3, 1).contiguous()
        x_mapped = self.W(x_mapped)
        
        # 2. Dynamic Message Passing using per-frame Inv-Distance Bipartite Graph
        # A shape: (B*T, V, V) | x_mapped shape: (B*T, V, out_channels)
        x_mapped = x_mapped.view(B*T, V, -1)
        out = torch.bmm(A, x_mapped) 
        
        # 3. Reshape and BN
        out = out.view(B, T, V, -1).permute(0, 3, 1, 2).contiguous() 
        out = self.bn(out)
        return self.relu(out)

class GCNBlock(nn.Module):
    """
    GCN Baseline Block: Spatial Graph Convolution ONLY.
    NO Temporal Convolution is applied.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNBlock, self).__init__()
        self.gcn = SpatialGCN(in_channels, out_channels)
        
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        return F.relu(x + res)

class GCNEncoder(nn.Module):
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_joints=15, num_rays=540):
        super(GCNEncoder, self).__init__()
        self.num_joints = num_joints
        self.num_rays = num_rays
        
        # Heterogeneous Linear Projections
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.ray_proj = nn.Linear(ray_in_dim, hidden_dim)
        
        # Bipartite Graph preserved for Spatial Mixing ablation
        self.graph_adj = AdaptiveGraphAdjacency(num_joints=num_joints, num_rays=num_rays)
        
        # Replacing TGCNBlocks with pure GCNBlocks
        self.gcn_blocks = nn.ModuleList([
            GCNBlock(hidden_dim, hidden_dim),
            GCNBlock(hidden_dim, hidden_dim*2)
        ])
        
        encoded_dim = hidden_dim * 2
        
        self.mu_net = nn.Linear(encoded_dim, z_dim)
        self.logvar_net = nn.Linear(encoded_dim, z_dim)
        
    def forward(self, node_feats, ray_feats):
        B, T, _, _ = node_feats.shape
        
        # [Fallback Safety Check]
        if ray_feats.shape[-1] == 3:
            ray_dist = torch.norm(ray_feats, p=2, dim=-1, keepdim=True)
            ray_feats = torch.cat([ray_feats, ray_dist], dim=-1)
            
        node_xyz = node_feats[..., :3]  
        ray_xyz = ray_feats[..., :3]    
        
        A = self.graph_adj(node_xyz, ray_xyz)
        
        h_nodes = self.node_proj(node_feats) 
        h_rays = self.ray_proj(ray_feats)    
        
        x = torch.cat([h_nodes, h_rays], dim=2) 
        x = x.permute(0, 3, 1, 2).contiguous()  
        
        # Pure Spatial Mixing (No Temporal Logic)
        for layer in self.gcn_blocks:
            x = layer(x, A)
            
        encoded_rays = x[:, :, :, self.num_joints:] 
        encoded_rays = encoded_rays.permute(0, 2, 3, 1).contiguous()
            
        scene_feat = x.mean(dim=-1).permute(0, 2, 1).contiguous()
        
        mu = self.mu_net(scene_feat)
        logvar = self.logvar_net(scene_feat)
        
        return mu, logvar, encoded_rays

class MoEDecoder(nn.Module):
    # Identical exactly to proposed
    def __init__(self, in_channels, out_dim=6, num_experts=3):
        super(MoEDecoder, self).__init__()
        self.num_experts = num_experts
        hidden_dim = in_channels // 2
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_experts), nn.Softmax(dim=-1))
    def forward(self, z_c):
        gate_weights = self.gate(z_c) 
        expert_outputs = torch.stack([expert(z_c) for expert in self.experts], dim=-1) 
        return torch.relu(torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1))

class IIWGCN_cVAE(nn.Module):
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_body_parts=6, num_experts=3):
        super(IIWGCN_cVAE, self).__init__()
        self.encoder = GCNEncoder(node_in_dim, ray_in_dim, hidden_dim, z_dim)
        self.decoder = MoEDecoder(in_channels=z_dim + (hidden_dim * 2), out_dim=num_body_parts, num_experts=num_experts)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, node_feats, ray_feats):
        B, T, _, _ = node_feats.shape
        num_rays = ray_feats.shape[2]
        mu, logvar, encoded_rays = self.encoder(node_feats, ray_feats)
        z = self.reparameterize(mu, logvar) if self.training else mu 
        z_expanded = z.unsqueeze(2).expand(B, T, num_rays, -1)
        z_c = torch.cat([z_expanded, encoded_rays], dim=-1) 
        pred_iiw = self.decoder(z_c).permute(0, 1, 3, 2).contiguous()
        return pred_iiw, mu, logvar
