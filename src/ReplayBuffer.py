import numpy as np
import torch
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        # We store the "raw" data in a deque
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        In Pong, 'state' is the 4-frame stack. 
        To be efficient, we only store the NEWEST frame here, 
        but for your first version, storing the stack is easier to code.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 1. Randomly pick experiences
        batch = random.sample(self.buffer, batch_size)
        
        # 2. Unpack them
        states, actions, rewards, next_states, dones = zip(*batch)

        # 3. Convert to NumPy then to Torch Tensors
        # This is the most efficient way to get data onto the M4 GPU
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).unsqueeze(1).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float()

        

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)