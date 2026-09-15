import os

import numpy as np

import CybORG.Agents as Agents
import Agents.LANCER.utils_random_sis_c as utils_random
import Agents.LANCER.CleanRLAgent as CleanRLAgent

class LANCER_SIS_C_v1( Agents.BaseAgent ):
    def __init__( self ):
        dirname = os.path.dirname( __file__ )
        # checkpoint = os.path.join( dirname, 'models', 'high_recall_ckpt.pth' )
        # checkpoint = os.path.join( dirname, 'models', 'final_sisc.pth' )
        # checkpoint = os.path.join( dirname, 'models', 'final_low_detect.pth' )
        checkpoint = os.path.join( dirname, 'models', 'final_new_redpill.pth' )
        self.agent = CleanRLAgent.CleanRLCheckpointAgent( checkpoint )
        self.count = None
        self.action_mapping_dict = {}
        self.map_action = utils_random.ActionMappingRandom()
        self.blue_table = utils_random.BlueTableRandom()
        self.last_action = None


    def train(self, results):
        pass


    def get_action(self, observation, action_space=None, true_obs=None):
        action_count = 242
        if isinstance( observation, np.ndarray ):
            numeric_action = self.agent.get_action( observation, action_count )
            return numeric_action

        if self.count is None:
            self.count = 1
            blue_info = self.blue_table._process_initial_obs(observation)
            vector = self.blue_table.observation_change(observation, self.last_action, baseline = True)
            true_obs = observation
            if true_obs is not None and action_space is not None:
                self.action_mapping_dict = self.map_action.action_mapping(action_space, true_obs)
        else:
            self.count += 1
            vector = self.blue_table.observation_change(observation, self.last_action)

        numeric_action = self.agent.get_action( vector, action_count )
        mapped_action = self.action_mapping_dict[int(numeric_action)]
        action = utils_random.numeric_to_cyborg_action(mapped_action)
        self.last_action = action
        self.last_numeric_action = numeric_action
        return action


    def end_episode(self):
        self.count = None
        self.action_mapping_dict = {}
        self.last_action = None


    def set_initial_values(self, observation = None, action_space = None, action_mapping_dict = None):
        self.action_mapping_dict = action_mapping_dict
