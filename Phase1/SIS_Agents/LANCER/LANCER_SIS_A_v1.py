import os

import numpy as np

import CybORG.Agents as Agents
import Agents.LANCER.utils_random_sis_a as utils_random
import Agents.LANCER.CleanRLAgent as CleanRLAgent

class LANCER_SIS_A_v1( Agents.BaseAgent ):
    def __init__( self ):
        dirname = os.path.dirname( __file__ )
        # checkpoint = os.path.join( dirname, 'models', 'sis-a_0.5_6000000_2025-02-12_12_07_42.pth' )
        # checkpoint = os.path.join( dirname, 'models', 'sis-a_8000000_2025-02-17_09_17_47.pth' )
        # checkpoint = os.path.join( dirname, 'models', 'sis-a_12000000_2025-02-17_18_07_43.pth' )
        # checkpoint = os.path.join( dirname, 'models', 'sis-a_new_red_pills_3600000_2025-02-20_09_13_09.pth' )
        checkpoint = os.path.join( dirname, 'models', 'sis-a_finetuned_newblinebugfix_18000000_2025-03-16_21_19_01.pth' )
        self.agent = CleanRLAgent.CleanRLCheckpointAgent( checkpoint )
        self.count = None
        self.action_mapping_dict = {}
        self.map_action = utils_random.ActionMappingRandom()
        self.blue_table = utils_random.BlueTableRandom()
        self.last_action = None


    def train(self, results):
        pass


    def get_action(self, observation, action_space=None, true_obs=None):
        action_count = 222
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
