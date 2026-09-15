import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

import CybORG.Agents as Agents

def create_network(input_size, output_size, default_architecture=True, network_size=64):
    if default_architecture:
        return nn.Sequential(
            layer_init(nn.Linear(input_size, network_size)),
            nn.Tanh(),
            layer_init(nn.Linear(network_size, network_size)),
            nn.Tanh(),
            layer_init(nn.Linear(network_size, output_size), std=1.0),
        )
    else: #network architecture used for SB3 implementation
        return nn.Sequential(
            nn.Linear(input_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, output_size),
        )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def linear( input_dim, output_dim, std=np.sqrt(2), bias=0.0 ):
    layer = nn.Linear(input_dim, output_dim)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

class BoostStyle( nn.Module ):
    def __init__(self, input_dim, action_space_size):
        super(BoostStyle, self).__init__()
        size = 64
        self.net1 = nn.Sequential(
            linear(input_dim, size),
            nn.Tanh(),
            linear(size, size),
            nn.Tanh(),
            linear(size, action_space_size, std=0.01)
        )
        self.net2 = linear(input_dim, action_space_size, std=0.01)

    def get_action_and_value(self, input, action=None):
        logits = self.net1(input) + self.net2(input)
        return F.softmax(logits, dim=-1)

    def get_action(self, x, action_space=None):
        if isinstance(x, tuple):
            obs, _ = x
        else:
            obs = x
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        logits = self.net1(obs_tensor) + self.net2(obs_tensor)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action


class CoreAgent( nn.Module ):
    def __init__( self, obs_size, action_space_size, device, include_critic ):
        super().__init__()
        if include_critic:
            self.critic = create_network( obs_size, 1 )
        self.actor = create_network( obs_size, action_space_size )
        self.device = device

    def get_action( self, obs, action_space ):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.actor(obs_tensor)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action


class Cage2StyleAgent(nn.Module):
    def __init__(self, obs_size, action_space_size, device ):
        super().__init__()
        default_architecture = False
        network_size = 256
        self.actor = create_network( obs_size, action_space_size, default_architecture, network_size )
        self.device = device

    def get_action(self, x, action_space=None, if_greedy=False):
        if isinstance(x, tuple):
            obs, _ = x
        else:
            obs = x
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.actor(obs_tensor)
        probs = Categorical(logits=logits)
        if if_greedy:
            action = probs.mode
        else:
            action = probs.sample()
        return action


class CleanRLCheckpointAgent( Agents.BaseAgent ):
    def __init__( self, checkpoint_path, device=None ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.agent = None
        self.device = device or torch.device( "cpu" )


    def get_action( self, x, action_space ):
        if isinstance( x, tuple ):
            obs, _ = x
        else:
            obs = x
        # assert obs is the ChallengeWrapper - "Blue" observations
        self.initialize_agent( obs, action_space )
        return self.agent.get_action( obs, action_space )


    def initialize_agent( self, obs, action_space ):
        if None != self.agent:
            return
        model_possibilities = [
            BoostStyle( len( obs ), action_space ),
            CoreAgent( len( obs ), action_space, self.device, True ),
            CoreAgent( len( obs ), action_space, self.device, False ),
            # Cage2StyleAgent( env_dict, network_size, default_architecture ),
            Cage2StyleAgent( len( obs ), action_space, self.device ),
        ]
        something_worked = False
        exns = []
        for model in model_possibilities:
            if something_worked:
                break
            try:
                self.agent = model
                model_state = torch.load( self.checkpoint_path, map_location=self.device )
                self.agent.load_state_dict( model_state )
                something_worked = True
            except Exception as exn:
                exns.append( exn )
        if not something_worked:
            print( "Failed to load checkpoint" )
            for exn in exns:
                print( exn )
            raise "Failed to load checkpoint"


    def end_episode( self ):
        pass


class CleanRLAgent( Agents.BaseAgent ):
    def __init__(self, agent:nn.Module, if_greedy=False):
        super().__init__()
        self.agent = agent
        self.if_greedy = if_greedy

    def get_action(self, x, action_space=None):
        return self.agent.get_action(x, if_greedy=self.if_greedy)
    
    def to(self, device):
        self.agent.to(device)

    def end_episode(self):
        """Allows an agent to update its internal state"""
        pass

    def set_initial_values(self, action_space, observation):
        pass
