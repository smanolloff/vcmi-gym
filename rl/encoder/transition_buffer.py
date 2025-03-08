import torch


class TransitionBuffer:
    def __init__(self, capacity, state_shape, action_dim, device="cpu"):
        self.capacity = capacity
        self.device = device

        self.state_buffer = torch.empty((capacity, *state_shape), dtype=torch.float32, device=device)
        self.mask_buffer = torch.empty((capacity, action_dim), dtype=torch.bool, device=device)
        self.done_buffer = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.action_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.reward_buffer = torch.empty((capacity,), dtype=torch.float32, device=device)

        self.index = 0
        self.full = False

    # Using compact version with single state and mask buffers
    # def add(self, state, action_mask, done, action, reward, next_state, next_action_mask, next_done):
    def add(self, state, action_mask, done, action, reward):
        self.state_buffer[self.index] = torch.as_tensor(state, dtype=torch.float32)
        self.mask_buffer[self.index] = torch.as_tensor(action_mask, dtype=torch.bool)
        self.done_buffer[self.index] = torch.as_tensor(done, dtype=torch.bool)
        self.action_buffer[self.index] = torch.as_tensor(action, dtype=torch.float32)
        self.reward_buffer[self.index] = torch.as_tensor(reward, dtype=torch.float32)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Get valid indices where done=False (episode not ended)
        valid_indices = torch.nonzero(~self.done_buffer[:max_index - 1], as_tuple=True)[0]
        sampled_indices = valid_indices[torch.randint(len(valid_indices), (batch_size,))]

        state = self.state_buffer[sampled_indices]
        # action_mask = self.mask_buffer[sampled_indices]
        action = self.action_buffer[sampled_indices]
        reward = self.reward_buffer[sampled_indices]
        next_state = self.state_buffer[sampled_indices + 1]
        next_action_mask = self.mask_buffer[sampled_indices + 1]
        next_done = self.done_buffer[sampled_indices + 1]

        return state, action, reward, next_state, next_action_mask, next_done

    def sample_iter(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Get valid indices where done=False
        valid_indices = torch.nonzero(~self.done_buffer[:max_index - 1], as_tuple=True)[0]
        shuffled_indices = valid_indices[torch.randperm(len(valid_indices))]

        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            yield (
                self.state_buffer[batch_indices],
                self.action_buffer[batch_indices],
                self.reward_buffer[batch_indices],
                self.state_buffer[batch_indices + 1],
                self.mask_buffer[batch_indices + 1],
                self.done_buffer[batch_indices + 1]
            )
