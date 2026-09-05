#model.py
"""
CNN architecture for FPG estimation from rPPG TF-Maps.
 
Description: two 2D conv blocks with ReLU + spatial downsampling, flattened features, 
a dense layer with dropout, and a linear regression head trained with L1 (MAE) loss.

"""

from torch import nn
 
 
class TinyCNN(nn.Module):
    def __init__(self, in_ch, p_drop=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(16, 32, 3, padding=1),   nn.ReLU(), nn.MaxPool2d(2),   # 32->16
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 1),
        )
 
    def forward(self, x):
        return self.net(x)
 