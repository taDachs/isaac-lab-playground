import skrl.agents.torch.simba_v2 as simba
import gymnasium as gym

import numpy as torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.utils.spaces.torch import compute_space_size
from skrl.envs.wrappers.torch.base import Wrapper

class VecNormalizeReward(Wrapper):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has
        an approximately fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.reset()

        self.return_rms = simba.TorchRunningMeanStd(size=())
        self.reward: torch.Tensor = torch.tensor([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self.g_max = g_max
        self._update_running_mean = True
        self.max_return: torch.Tensor = torch.tensor([0.0])

    def reset(self):
        return self.env.reset()

    @torch.no_grad()
    def step(self, action) -> tuple:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(action)
        print(reward)

        self.reward = self.reward * self.gamma + reward
        if self._update_running_mean:
            self.return_rms.update(self.reward)

        self.max_return = torch.max(self.max_return, self.reward.abs().max())

        var_denominator = torch.sqrt(self.return_rms.var + self.epsilon)
        min_required_denominator = self.max_return / self.g_max
        denominator = torch.max(var_denominator, min_required_denominator)

        normalized_reward = reward / denominator

        if terminated or truncated:
            self.reward = torch.tensor([0.0])

        print(normalized_reward)

        return obs, normalized_reward, terminated, truncated, info

    def state(self):
        return self.env.state()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

