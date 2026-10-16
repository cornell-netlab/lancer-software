# Example submission.py
import os
import torch
import pickle
import numpy as np
from gym import Space

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from ray.rllib.policy.policy import PolicySpec, Policy

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gymnasium.spaces import MultiDiscrete
from ray.rllib.models import ModelCatalog

from CustomRLLib.Models.TorchActionMaskModel import TorchActionMaskModel
from MaskWrapper import MaskWrapper

ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

class Agent(BaseAgent):
    def __init__(self, name: str = None, policy = None):
        super().__init__(name)
        self.policy = policy

    def convert_obs(self, obs):
        """
        For models trained with RLLib using MultiDiscrete Space
        """
        bit_obs_array = []
        if obs[0] == 0:
            bit_obs_array += [1, 0, 0]
        elif obs[0] == 1:
            bit_obs_array += [0, 1, 0]
        else:
            bit_obs_array += [0, 0, 1]

        for i in range(1, len(obs)):
            if obs[i] == 0:
                bit_obs_array += [1, 0]
            else:
                bit_obs_array += [0, 1]
        return np.array(bit_obs_array)

    def get_action(self, observation: dict, action_space: Space):
        obs = {
            'observations': self.convert_obs(observation['observations']),
            'action_mask': observation['action_mask'],
            'floats': observation['floats']
        }
        result = self.policy.compute_single_action(obs)
        return result[0]
    
class Submission:

    # Submission name
    NAME: str = "PPO-init"

    # Name of your team
    TEAM: str = "LANCER"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "PPO"

    # Use this function to define your agents.
    AGENTS = {
    f"blue_agent_{agent}": Agent(f'Agent{agent}', Policy.from_checkpoint(os.path.join(os.path.join(os.path.dirname(__file__), 'staging/policies'), f'Agent{agent}/'))) for agent in range(5)
    }

    #exe_dir = os.getcwd()
    #policy_dir = os.path.join(exe_dir, 'staging/policies')

    #AGENTS: dict[str, BaseAgent] = {
    #    f"blue_agent_{agent}": Agent(f'Agent{agent}', Policy.from_checkpoint(os.path.dirname(f'{policy_dir}/Agent{agent}/'))) for agent in range(5)
    #}

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return MaskWrapper(env, mask_training=True, evaluation=True)
