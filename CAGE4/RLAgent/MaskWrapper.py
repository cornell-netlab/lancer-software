from __future__ import annotations
from typing import Any

import numpy as np

from CybORG.Agents.Wrappers import BlueEnterpriseWrapper, EnterpriseMAE
import BlueFixedActionWrapper
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG import CybORG

import functools
import itertools

from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (
    BlueFixedActionWrapper,
    MESSAGE_LENGTH,
    EMPTY_MESSAGE,
    NUM_MESSAGES,
)

from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

NUM_SUBNETS = 9
NUM_HQ_SUBNETS = 3

MAX_USER_HOSTS = EnterpriseScenarioGenerator.MAX_USER_HOSTS
MAX_SERVER_HOSTS = EnterpriseScenarioGenerator.MAX_SERVER_HOSTS
MAX_HOSTS = MAX_USER_HOSTS + MAX_SERVER_HOSTS

from gymnasium import Space, spaces

from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from BlueRewardMachine import get_reward_list


def action_filter(action):
  return "DeployDecoy" in action or "Analyse" in action #or "AllowTrafficZone" in action or "BlockTrafficZone" in action or "Monitor" in action
  

sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent, red_agent_class=FiniteStateRedAgent, steps=500)
cy = CybORG(scenario_generator=sg, seed=1234)
mae = EnterpriseMAE(cy)
action_masks = {agent: [action_filter(str(action)) for action in mae.get_action_space(agent)['labels']] for agent in mae.agents}

monitor_masks = {agent: ["Monitor" in str(action) for action in mae.get_action_space(agent)['labels']] for agent in mae.agents}

restore_masks = {agent: ["Restore" in str(action) for action in mae.get_action_space(agent)['labels']] for agent in mae.agents}

action_names = {agent: [str(action) for action in mae.get_action_space(agent)['labels']] for agent in mae.agents}

agent_subnets = {agent: set(str(action).split(" ")[1].split("subnet")[0]+"subnet" for action in mae.get_action_space(agent)['labels'] if "Restore" in str(action)) for agent in mae.agents}

agent_hosts = {agent: sorted([a.split(' ')[1] for a in mae.get_action_space(agent)['labels'] if "Restore" in a]) for agent in mae.agents}

class MaskWrapper(EnterpriseMAE):
    """A wrapper designed to support CAGE Challenge 4 (RLlib Compatible).

    Creates a vector output for a neural network by directly pulling
    information out of the state object.
    """

    def __init__(self, env, mask_training=True, evaluation=False, *args, **kwargs):
        # super(BlueEnterpriseWrapper, self).__init__(env, *args, **kwargs)
        # super(MultiAgentEnv, self).__init__()
        super().__init__(env, *args, **kwargs)
        self.mask_training=mask_training
        self.evaluation = evaluation
        self.step_count = 0

    def _get_init_obs_spaces(self):
        """Calculates the size of the largest observation space for each agent."""
        short_observation_space, long_observation_space = super()._get_init_obs_spaces()
        # Extend the observation space by 1

        self._observation_space = spaces.Dict({
            agent: spaces.Dict({
                "observations": long_observation_space,
                "action_mask": spaces.MultiBinary(self.action_space(agent).n),
                "floats": spaces.Box(low=0, high=1, shape=(1+(6+10)*3,))
            })
            if self.is_padded or agent == "blue_agent_4"
            else spaces.Dict({
                "observations": short_observation_space,
                "action_mask": spaces.MultiBinary(self.action_space(agent).n),
                "floats": spaces.Box(low=0, high=1, shape=(1+6+10,))
            })
            for agent in self.agents
        })
        return short_observation_space, long_observation_space

    def step(
        self,
        action_dict: dict[str, Any] | None = None,
        messages: dict[str, Any] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        
        for agent in self.agents:
            for host in self.last_reset[agent]:
                self.last_reset[agent][host] += 1

        # updated action_dict for "Analyse" and "DeployDecoy"
        for agent in self.agents:
            obs = self.env.get_observation(agent)
            if 'action' in obs:
                action = str(obs['action']).split(' ')
                if len(action) == 2 and action[0] == 'Analyse':
                    host = action[1]
                    labels = self.get_action_space(agent)['labels']
                    if 'escalate.sh' in str(obs):
                        action_dict[agent] = labels.index(f"Restore {host}")
                    elif 'cmd.sh' in str(obs):
                        action_dict[agent] = labels.index(f"Remove {host}")
                    self.last_reset[agent][host] = 0
                        
        obs, rew, terminated, truncated, info = super().step(
            action_dict=action_dict, messages=messages
        )
       
        self.step_count += 1
        if not self.evaluation:
            reward_list = get_reward_list()
            def is_valid_restore(agent,action):
                if self.action_mask(agent)[action] == False:
                    return False
                return "Restore" in str(self.get_action_space(agent)['labels'][action])
            
            restores = {a: -1 if is_valid_restore(a, action_dict[a]) else 0 for a in self.agents}
            contractor_reward = sum([r for subnet, r in reward_list if subnet == "contractor_network_subnet"])
            rewards = {a: sum([r for subnet, r in reward_list if subnet in agent_subnets[a]]) for a in self.agents}

            # normalize the reward
            agent_rewards = {a: (restores[a] + rewards[a]) for a in self.agents}
            #print(agent_rewards)
        
            sum_rewards = sum(agent_rewards.values()) + contractor_reward
            # print(f"{reward_list=}\n{restores=}\n{rewards=}\n{agent_rewards=}\n{rew=}\n{sum_rewards=}\n-----------\n")
            # print(action_dict)
            # print({a: self.get_action_space(a)['actions'][action_dict[a]] for a in self.agents})
            # print({a: restore_masks[a][action_dict[a]] for a in self.agents})
            #assert sum_rewards >= rew['blue_agent_0']
            #assert sum_rewards == rew['blue_agent_0']
            obs = self.create_obs(obs)

            assert isinstance(obs, dict)
            return obs, agent_rewards, terminated, truncated, info
        else:
            obs = self.create_obs(obs)
        return obs, rew, terminated, truncated, info

    def create_obs(self, obs):
        last_reset_vecs = {}
        for agent in self.agents:
            hosts = agent_hosts[agent]
            last_reset_vecs[agent] = [self.last_reset[agent][host]/500 for host in hosts]

        obs = ({a: {
            "observations": obs[a],
            "floats": np.array([self.step_count/500] + last_reset_vecs[a]),
            "action_mask": 1*np.array(self.custom_action_mask(a)) if self.mask_training else np.ones(len(self.custom_action_mask(a)))
        } for a in self.agents})
        return obs

    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and update the observation space.

        Args: All arguments are forwarded to the env provided to __init__.

        Returns
        -------
        observation : dict[str, Any]
            The observations corresponding to each agent, translated into a vector format.
        info : dict[str, dict]
            Forwarded from self.env.
        """
        observations, info = super().reset(*args, **kwargs)

        self.last_reset = {}
        for agent in self.agents:
            hosts = agent_hosts[agent]
            self.last_reset[agent] = {host: 0 for host in hosts}

        obs = self.create_obs(observations)
        # ({
        #     a: {"observations": observations[a],
        #         "action_mask": 1*np.array(self.custom_action_mask(a)) if self.mask_training else np.ones(len(self.custom_action_mask(a)))}
        #     for a in self.agents
        # })
        assert isinstance(obs, dict)
        # assert info.keys() == obs.keys()
        self.step_count = 0
        return obs, info

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name: str) -> Space | spaces.Dict:
        """Returns the multi-discrete space corresponding to the given agent."""
        return self._observation_space[agent_name]

    @functools.lru_cache(maxsize=None)
    def observation_spaces(self) -> dict[str, Space | spaces.Dict]:
        """Returns multi-discrete spaces corresponding to each agent."""
        return {a: self.observation_space(a) for a in self.possible_agents}

    def custom_action_mask(self, agent_name: str) -> list[bool]:
        # Invalid action masking
        mask = self.action_mask(agent_name)
        # Only do remove/restore/decoy
        mask = [mask[i] and action_masks[agent_name][i] for i in range(len(mask))]
        # Monitor while an action is in progress
        obs = self.env.get_observation(agent_name)
        if "PROGRESS" in str(obs):
            mask = monitor_masks[agent_name]
        return mask
    
