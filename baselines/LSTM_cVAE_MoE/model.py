import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# To import utils regardless of running directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))

class LSTMEncoder(nn.Module):
    """
    LSTM Baseline: Flattens spatial dependencies and models the sequence 
    purely using Recurrent Neural Networks (LSTM) over the time dimension.
    """
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_joints=15, num_rays=540):
        super(LSTMEncoder, self).__init__()
        self.num_joints = num_joints
        self.num_rays = num_rays
        
        # Spatial dimensions are completely flattened for the LSTM
        flat_dim = (num_joints * node_in_dim) + (num_rays * ray_in_dim)
        encoded_dim = hidden_dim * 2
        
        self.lstm = nn.LSTM(
            input_size=flat_dim,
            hidden_size=encoded_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True # Outputs 2 * encoded_dim
        )
        
        # Ray specific encoder to provide spatial MoE condition
        self.ray_mlp = nn.Sequential(
            nn.Linear(ray_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim),
            nn.ReLU()
        )
        
        # Bidirectional reduces to encoded_dim for the Latent Space
        self.reduction = nn.Linear(encoded_dim * 2, encoded_dim)
        
        self.mu_net = nn.Linear(encoded_dim, z_dim)
        self.logvar_net = nn.Linear(encoded_dim, z_dim)
        
    def forward(self, node_feats, ray_feats):
        B, T, _, _ = node_feats.shape
        
        # [Fallback Safety Check]
        if ray_feats.shape[-1] == 3:
            ray_dist = torch.norm(ray_feats, p=2, dim=-1, keepdim=True)
            ray_feats = torch.cat([ray_feats, ray_dist], dim=-1)
            
        # 1. Spatially Flatten the nodes and rays for recurrent LSTM processing
        N_flat = node_feats.view(B, T, -1)
        R_flat = ray_feats.view(B, T, -1)
        
        x = torch.cat([N_flat, R_flat], dim=-1) # (B, T, flat_dim)
        
        # 2. Recurrent Time Sequence Modeling
        lstm_out, _ = self.lstm(x) # (B, T, encoded_dim * 2)
        scene_feat = torch.relu(self.reduction(lstm_out)) # (B, T, encoded_dim)
        
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

class IIWLSTM_cVAE(nn.Module):
    def __init__(self, node_in_dim=9, ray_in_dim=4, hidden_dim=64, z_dim=32, num_body_parts=6, num_experts=3):
        super(IIWLSTM_cVAE, self).__init__()
        self.encoder = LSTMEncoder(node_in_dim, ray_in_dim, hidden_dim, z_dim)
        
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
