import torch.nn as nn
import torch
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):

        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        if len(self.priorities) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]
        importance_weights = (1 / (len(self.buffer) * probabilities[indices])) ** beta
        importance_weights = importance_weights / importance_weights.max()
        return batch, indices, torch.FloatTensor(importance_weights).cuda()

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = abs(error) + 1e-5

    def __len__(self):
        return len(self.buffer)
