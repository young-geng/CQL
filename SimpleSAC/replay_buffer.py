from copy import copy, deepcopy
from queue import Queue
import threading

import d4rl

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...]
        )


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return dict(
        observations=batch['observations'][indices, ...],
        actions=batch['actions'][indices, ...],
        rewards=batch['rewards'][indices, ...],
        next_observations=batch['next_observations'][indices, ...],
        dones=batch['dones'][indices, ...]
    )

def concatenate_batches(batches):
    return dict(
        observations=np.concatenate([batch['observations'] for batch in batches], axis=0).astype(np.float32),
        actions=np.concatenate([batch['actions'] for batch in batches], axis=0).astype(np.float32),
        rewards=np.concatenate([batch['rewards'] for batch in batches], axis=0).astype(np.float32),
        next_observations=np.concatenate([batch['next_observations'] for batch in batches], axis=0).astype(np.float32),
        dones=np.concatenate([batch['dones'] for batch in batches], axis=0).astype(np.float32),
    )


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )
