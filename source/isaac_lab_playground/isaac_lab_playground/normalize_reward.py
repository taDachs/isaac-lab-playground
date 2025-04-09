from typing import Tuple, Any, Union
import skrl.agents.torch.simba_v2 as simba
import gymnasium as gym

import numpy as torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.utils.spaces.torch import compute_space_size
from skrl.envs.wrappers.torch.base import Wrapper

class NormalizeReward(Wrapper):
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

        self.return_rms = simba.TorchRunningMeanStd(size=(), device=self._device)
        self.reward = torch.zeros((self.num_envs, 1), device=self._device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.g_max = g_max
        self._update_running_mean = True
        self.max_return: torch.Tensor = torch.zeros(1, device=self._device)

    @torch.no_grad()
    def step(self, action) -> tuple:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = self._env.step(action)

        self.reward = self.reward * self.gamma + reward
        if self._update_running_mean:
            self.return_rms.forward(self.reward, train=True)

        self.max_return = torch.max(self.max_return, self.reward.abs().max())

        var_denominator = torch.sqrt(self.return_rms.running_variance + self.epsilon)
        min_required_denominator = self.max_return / self.g_max
        denominator = torch.max(var_denominator, min_required_denominator)

        normalized_reward = reward / denominator

        self.reward *= (terminated | truncated).logical_not().double()

        return obs, normalized_reward, terminated, truncated, info

    @property
    def state_space(self) -> Union[gym.Space, None]:
        """State space"""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gym.Space:
        """Observation space"""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space["policy"]

    @property
    def action_space(self) -> gym.Space:
        """Action space"""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        return self._env.reset()

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
