import random
from collections import deque
import numpy as np
import time
import torch


class ReplayBuffer:
    def __init__(self, max_size, model):
        self.buffer = deque(maxlen=max_size)
        self.model = model

    def add(self, action, cut_states, cut_actions, cut_returns_to_go, attention_mask, cut_timesteps):
        self.buffer.append(
            (action, cut_states, cut_actions, cut_returns_to_go, attention_mask, cut_timesteps))

    def sample(self, batch_size):
        action, cut_states, cut_actions, cut_returns_to_go, attention_mask, cut_timesteps = zip(
            *self.buffer)
        action = torch.tensor(np.concatenate(action)).to(device=self.model.device)
        states = torch.stack(cut_states).reshape(batch_size, -1, self.model.config.state_dim)
        actions = torch.stack(cut_actions).reshape(batch_size, -1, self.model.config.act_dim).detach()
        returns_to_go = torch.stack(cut_returns_to_go).reshape(batch_size, -1, 1)
        attention_masks = torch.stack(attention_mask).reshape(batch_size, -1)
        timesteps = torch.stack(cut_timesteps).reshape(batch_size, -1)
        return action, states, actions, returns_to_go, attention_masks, timesteps

    def clear(self):
        print(len(self.buffer))
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
