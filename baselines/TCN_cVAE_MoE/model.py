import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# To import utils regardless of running directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))

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

class TCNEncoder(nn.Module):
    """
    TCN Baseline: Uses Dilated Temporal Convolutions to process time, 
    but provides ZERO spatial message passing between nodes. 
    (Each joint and ray is processed completely independently across time).
    """
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_joints=15, num_rays=540):
        super(TCNEncoder, self).__init__()
        self.num_joints = num_joints
        self.num_rays = num_rays
        
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.ray_proj = nn.Linear(ray_in_dim, hidden_dim)
        
        # Notice: No AdaptiveGraphAdjacency and no SpatialGCN blocks.
        self.tcn_blocks = nn.ModuleList([
            TemporalTCN(hidden_dim, hidden_dim, kernel_size=9, stride=1, dilation=1),
            TemporalTCN(hidden_dim, hidden_dim*2, kernel_size=5, stride=1, dilation=2)
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
            
        # 1. Project heterogeneous features
        h_nodes = self.node_proj(node_feats) # (B, T, 15, hidden)
        h_rays = self.ray_proj(ray_feats)    # (B, T, 540, hidden)
        
        x = torch.cat([h_nodes, h_rays], dim=2) 
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, hidden, T, 555)
        
        # 2. Pure Temporal Convolutions (No spatial mixing)
        for layer in self.tcn_blocks:
            x = layer(x) # x is updated independently per spatial node
            
        # 3. Extract Encoded Rays for MoE Condition
        encoded_rays = x[:, :, :, self.num_joints:] # (B, encoded_dim, T, 540)
        encoded_rays = encoded_rays.permute(0, 2, 3, 1).contiguous() # (B, T, 540, encoded_dim)
            
        # 4. Global spatial pooling over all independent nodes
        scene_feat = x.mean(dim=-1).permute(0, 2, 1).contiguous() # (B, T, encoded_dim)
        
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

class IIWTCN_cVAE(nn.Module):
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_body_parts=6, num_experts=3):
        super(IIWTCN_cVAE, self).__init__()
        self.encoder = TCNEncoder(node_in_dim, ray_in_dim, hidden_dim, z_dim)
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
