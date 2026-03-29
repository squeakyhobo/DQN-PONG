import torch.nn as nn
import torch




class Qnet(nn.Module):

    def __init__(self,n_actions):
        super().__init__()

        

        self.net = nn.Sequential(
        # Conv Layer 1: 4 channels in, 32 out, 8x8 kernel
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        
        # Conv Layer 2: 32 in, 64 out, 4x4 kernel
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        
        # Conv Layer 3: 64 in, 64 out, 3x3 kernel
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        
        # Flatten the 3D output for the Dense layers
        nn.Flatten(),
        
        # Fully Connected Layer 1
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        
        # Output Layer: Value for each action
        nn.Linear(512, n_actions)
        )

        
    
    def forward(self,x):
        logits = self.net(x)
        return logits
  


