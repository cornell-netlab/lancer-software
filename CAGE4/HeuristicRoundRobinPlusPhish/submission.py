from __future__ import annotations
from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from heuristic import Agent, config, MaskWrapper, Config

agents = {
    f"blue_agent_{agent}": Agent(f"blue_agent_{agent}", Config(config)) for agent in range(5)
}


class Submission:

    # Submission name
    NAME: str = "SUBMISSION NAME"

    # Name of your team
    TEAM: str = "TEAM NAME"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "TECHNIQUE NAME"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = agents

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG, *args, **kwargs) -> MultiAgentEnv:
        return MaskWrapper(env, agents)
