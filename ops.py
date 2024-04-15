import torch
from torch import random
import torch.nn as nn

class HyperSphereLoss(nn.Module):
    def forward(self, input):

        q = self.project_to_hypersphere(input) #[64, 3]
        q_norm = torch.norm(q, dim=1) ** 2     #[64]
        
        loss = (2 * q[:, -1]) / (1 + q_norm)   
        return torch.mean(torch.acos(loss))    


    def project_to_hypersphere(self, v):
        v_norm = torch.norm(v, dim=1, keepdim=True) ** 2 #[64, 1]
        a = 2 * v / (v_norm + 1)                         #[64, 2]
        b = (v_norm - 1) / (v_norm + 1)                  #[64, 1]

        return torch.cat([a, b], dim=1)

