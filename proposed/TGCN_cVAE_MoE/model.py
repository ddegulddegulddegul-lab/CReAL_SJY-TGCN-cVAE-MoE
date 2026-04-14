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
        out = torch.bmm(A, x_mapped) # bmm automatically handles batched matrix multiplication!
        
        # 3. Reshape and BN
        out = out.view(B, T, V, -1).permute(0, 3, 1, 2).contiguous() 
        out = self.bn(out)
        return self.relu(out)

class TemporalTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):
        super(TemporalTCN, self).__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(kernel_size, 1), 
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        return self.relu(out + res)

class TGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tcn_kernel_size=9, stride=1, dilation=1):
        super(TGCNBlock, self).__init__()
        self.gcn = SpatialGCN(in_channels, out_channels)
        self.tcn = TemporalTCN(out_channels, out_channels, kernel_size=tcn_kernel_size, stride=stride, dilation=dilation)
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return F.relu(x + res)

class TGCNEncoder(nn.Module):
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_joints=15, num_rays=540):
        # Note: ray_in_dim is now 4 (XYZ + L2 Distance)
        super(TGCNEncoder, self).__init__()
        self.num_joints = num_joints
        self.num_rays = num_rays
        
        # Heterogeneous Linear Projections into a common embedding space
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.ray_proj = nn.Linear(ray_in_dim, hidden_dim)
        
        self.graph_adj = AdaptiveGraphAdjacency(num_joints=num_joints, num_rays=num_rays)
        
        self.tgcn_blocks = nn.ModuleList([
            TGCNBlock(hidden_dim, hidden_dim, tcn_kernel_size=9, stride=1, dilation=1),
            TGCNBlock(hidden_dim, hidden_dim*2, tcn_kernel_size=5, stride=1, dilation=2)
        ])
        
        encoded_dim = hidden_dim * 2
        
        self.mu_net = nn.Linear(encoded_dim, z_dim)
        self.logvar_net = nn.Linear(encoded_dim, z_dim)
        
    def forward(self, node_feats, ray_feats):
        B, T, _, _ = node_feats.shape
        
        # [Fallback Safety Check] Ensure ray_feats has the 4D (XYZ + Distance) dimension
        if ray_feats.shape[-1] == 3:
            ray_dist = torch.norm(ray_feats, p=2, dim=-1, keepdim=True)
            ray_feats = torch.cat([ray_feats, ray_dist], dim=-1)
            
        # 1. Extract raw XYZ for Inverse Distance formulation before projection
        node_xyz = node_feats[..., :3]  # (B, T, 15, 3)
        ray_xyz = ray_feats[..., :3]    # (B, T, 540, 3)
        
        # Calculate dynamic Adjacency Matrix (B*T, 555, 555)
        A = self.graph_adj(node_xyz, ray_xyz)
        
        # 2. Project heterogeneous features into shared hidden_dim
        h_nodes = self.node_proj(node_feats) # (B, T, 15, hidden)
        h_rays = self.ray_proj(ray_feats)    # (B, T, 540, hidden)
        
        # Concatenate nodes: 15 joints + 540 rays = 555 Total Nodes
        x = torch.cat([h_nodes, h_rays], dim=2) 
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, hidden, T, 555)
        
        # 3. Spatio-Temporal Graph Convolutions
        for layer in self.tgcn_blocks:
            x = layer(x, A)
            
        # 4. Extract Encoded Rays for MoE Condition
        encoded_rays = x[:, :, :, self.num_joints:] # (B, encoded_dim, T, 540)
        encoded_rays = encoded_rays.permute(0, 2, 3, 1).contiguous() # (B, T, 540, encoded_dim)
            
        # 5. Global spatial pooling over all 555 nodes for latent Z representation
        scene_feat = x.mean(dim=-1).permute(0, 2, 1).contiguous() # (B, T, encoded_dim)
        
        mu = self.mu_net(scene_feat)
        logvar = self.logvar_net(scene_feat)
        
        return mu, logvar, encoded_rays

class MoEDecoder(nn.Module):
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
        self.last_gate_weights = gate_weights.detach()
        expert_outputs = torch.stack([expert(z_c) for expert in self.experts], dim=-1) 
        out = torch.sum(expert_outputs * gate_weights.unsqueeze(-2), dim=-1) 
        return torch.relu(out) 

class IIWTGCN_cVAE(nn.Module):
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_body_parts=6, num_experts=3):
        super(IIWTGCN_cVAE, self).__init__()
        self.encoder = TGCNEncoder(node_in_dim, ray_in_dim, hidden_dim, z_dim)
        
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
