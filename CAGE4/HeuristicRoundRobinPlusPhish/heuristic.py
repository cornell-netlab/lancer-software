#!/usr/bin/env python
# coding: utf-8

# # PPO for Cage 4

# In[17]:


import random
import numpy as np
import concurrent.futures
from collections import namedtuple
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent, KeyboardAgent
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Agents.Wrappers.TrueStateWrapper import TrueStateTableWrapper
import re
from statistics import mean, stdev
import time

try:
   from IPython import get_ipython
   IS_NOTEBOOK = 'IPKernelApp' in get_ipython().config
except Exception:
   IS_NOTEBOOK = False

if IS_NOTEBOOK:
   from CybORG.Shared.BlueRewardMachine import set_is_notebook
   set_is_notebook(IS_NOTEBOOK)

if IS_NOTEBOOK:
   class Config:
      def __init__(self, dict):
         if isinstance(dict, Config):
            dict = dict._data
         self._data = dict

      def __setattr__(self, key, value):
         if key == '_data':
            self.__dict__['_data'] = value
         else:
            self._data[key] = value

      def __getattr__(self, key):
         if key == '_data':
            return self.__dict__['_data']
         else:
            return self._data[key]
else:
   class Config:
      def __init__(self, dict):
         self.__dict__.update(dict)


# In[18]:


def setup_cyborg(steps=500, seed=None):
   sg = EnterpriseScenarioGenerator(
      blue_agent_class=SleepAgent, # use a stand-in agent that we will overwrite the actions of
      green_agent_class=EnterpriseGreenAgent,
      red_agent_class=FiniteStateRedAgent,
      steps=steps
   )
   return CybORG(scenario_generator=sg, seed=seed)

def run_sim(config):
   print(f"Starting run {config}")
   config = Config(config)
   cyborg = setup_cyborg(steps=config.steps)
   agents = {
       f"blue_agent_{agent}": Agent(f"blue_agent_{agent}", config) for agent in range(5)
   }
   wrapped = MaskWrapper(cyborg, agents)
   obs, _ = wrapped.reset()
   reward = 0
   for step in range(config.steps):
      actions = {
         agent: agents[agent].get_action(obs[agent], wrapped.action_space(agent)) for agent in agents
      }
      obs, rew, term, trunc, info = wrapped.step(actions)
      done = {
         agent: term.get(agent, False) or trunc.get(agent, False)
         for agent in wrapped.agents
      }
      if all(done.values()):
         break
      # print(rew.values())
      reward += mean(rew.values())
      # if step % 50 == 0:
         # print(f"{step}: {reward}")
   return reward

def run_par(function, inputs):
    """
    Executes a function in parallel on a set of inputs and yields (input, result) pairs as they complete.

    :param function: The function to execute in parallel. It should accept a single argument.
    :param inputs: An iterable of inputs to feed into the function.
    """
    if IS_NOTEBOOK:
        # Use a single process in a notebook
         for input_value in inputs:
            result = function(input_value)
            yield (input_value, result)
         return
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all the tasks and keep a dictionary of future to input
        future_to_input = {executor.submit(function, input_value): input_value for input_value in inputs}

        # Yield results as they complete
        for future in concurrent.futures.as_completed(future_to_input):
            input_value = future_to_input[future]
            try:
                result = future.result()
                yield (input_value, result)
            except Exception as exc:
                print(f"Exception for input {input_value}: {exc}")
                yield (input_value, None)


# In[19]:


agent_names = ["blue_agent_0", "blue_agent_1", "blue_agent_2", "blue_agent_3", "blue_agent_4"]

cyborg = setup_cyborg( seed=np.random.default_rng( 42 ) )
mae = EnterpriseMAE(cyborg)
mae.get_action_space(agent_names[0])
agent_action_names = {
   agent: [s.replace("[Invalid] ", "") for s in mae.get_action_space(agent)['labels']] for agent in agent_names
}
agent_action_types = {agent: [s.split()[0] for s in actions] for agent, actions in agent_action_names.items() for action in actions}
action_type_masks = {agent: {action_type2: [action_type2 == action_type for action_type in agent_action_types[agent]] for action_type2 in agent_action_types[agent]} for agent in agent_names}

action_types_short = {
   'Analyse': 'a',
   'Remove': 'r',
   'Restore': 'R',
   'DeployDecoy': 'D',
   'Monitor': 'M',
   'AllowTrafficZone': 'A',
   'BlockTrafficZone': 'B',
   'Sleep': 'S'
}
agent_action_ty = {agent: "".join([action_types_short[action] for action in actions]) for agent, actions in agent_action_types.items()}

action_durations = {
   'Analyse': 2,
   'Remove': 3,
   'Restore': 5,
   'DeployDecoy': 2,
   'Monitor': 1,
   'AllowTrafficZone': 1,
   'BlockTrafficZone': 1,
   'Sleep': 1
}
agent_action_durations = {agent: [action_durations[action] for action in actions] for agent, actions in agent_action_types.items()}

subnets = {
   'admin_network': 'd',
   'office_network': 'o',
   'operational_zone_a': 'a',
   'operational_zone_b': 'b',
   'public_access_zone': 'p',
   'restricted_zone_a': 'A',
   'restricted_zone_b': 'B'
 }
hosts = {
   action.split("subnet_")[1]: action.split("subnet_")[1][0] + action.split("subnet_")[1][-1]
   for action_names in agent_action_names.values()
   for action in action_names
   if "host" in action
}

def replaces(dict, str):
   for key in dict:
      str = str.replace(key, dict[key])
   return str

def shorten(action_name):
   action_name = replaces(subnets, action_name)
   action_name = replaces(hosts, action_name)
   action_name = replaces({a: s+"." for a,s in action_types_short.items()}, action_name)
   action_name = replaces({"_subnet_": "", " ":"", "_subnet": ""}, action_name)
   action_name = replaces({"contractor_network": "c", "internet": "i"}, action_name)
   # remove everything between parens with a regex (there can be multiple parens)
   action_name = re.sub(r'\([^)]*\)', '', action_name).replace("<-", "")
   return action_name

agent_action_n = {agent: [shorten(action) for action in actions] for agent, actions in agent_action_names.items()}

def short_actions(actions):
   return "".join([agent_action_n[agent][actions[agent]].ljust(6) for agent in agent_names])

action_space_sizes = {agent: len(actions) for agent,actions in agent_action_names.items()}
{agent: "".join(s) for agent,s in agent_action_n.items()}
agent_action_ty

def get_observation_description(agent):
    S = 9
    H = 16
    descr = [{'descr': "Mission phase", 'class': 'phase'}]
    if agent == 'blue_agent_4':
        n = 3
    else:
        n = 1
    for i in range(n):
        extra = f" ({i})" if n > 1 else ""
        for s in range(S):
            descr.append({'descr': f"Current subnet is {s} {extra}", 'class': 'subnet'})
        for s in range(S):
            descr.append({'descr': f"Subnet {s} is blocked {extra}", 'class': 'blocked'})
        for s in range(S):
            descr.append({'descr': f"Subnet {s} should be blocked {extra}", 'class': 'should-block'})
        for h in range(H):
            us = "server" if h < 6 else "user"
            h2 = h if h < 6 else h - 6
            descr.append({'descr': f"Malicious process detected on {us} host {h2} {extra}", 'class': 'malicious-process'})
        for h in range(H):
            us = "server" if h < 6 else "user"
            h2 = h if h < 6 else h - 6
            descr.append({'descr': f"Malicious connection detected on {us} host {h2} {extra}", 'class': 'malicious-connection'})
    for i in range(4):
        for k in range(8):
            descr.append({'descr': f"Message {i} bit {k}", 'class': 'message-bit'})
    return descr

agent_observation_lengths = {agent: len(get_observation_description(agent)) for agent in agent_names}
agent_subnet_idx = {agent: [i for i in range(agent_observation_lengths[agent]) if get_observation_description(agent)[i]['class'] == 'subnet'] for agent in agent_names}
agent_subnet_status_idx = {agent: [i for i in range(agent_observation_lengths[agent]) if get_observation_description(agent)[i]['class'] == 'blocked'] for agent in agent_names}
agent_subnet_should_idx = {agent: [i for i in range(agent_observation_lengths[agent]) if get_observation_description(agent)[i]['class'] == 'should-block'] for agent in agent_names}
agent_block_subnet_idx = {agent: [i for i in range(action_space_sizes[agent]) if agent_action_ty[agent][i] == 'B'] for agent in agent_names}
agent_allow_subnet_idx = {agent: [i for i in range(action_space_sizes[agent]) if agent_action_ty[agent][i] == 'A'] for agent in agent_names}

agent_malicious_process_idx = {agent: [i for i in range(agent_observation_lengths[agent]) if get_observation_description(agent)[i]['class'] == 'malicious-process'] for agent in agent_names}
agent_malicious_connection_idx = {agent: [i for i in range(agent_observation_lengths[agent]) if get_observation_description(agent)[i]['class'] == 'malicious-connection'] for agent in agent_names}

agent_restore_idx = {agent: [i for i in range(action_space_sizes[agent]) if agent_action_ty[agent][i] == 'R'] for agent in agent_names}
agent_remove_idx = {agent: [i for i in range(action_space_sizes[agent]) if agent_action_ty[agent][i] == 'r'] for agent in agent_names}
agent_analyse_idx = {agent: [i for i in range(action_space_sizes[agent]) if agent_action_ty[agent][i] == 'a'] for agent in agent_names}


# In[20]:


agent_action_names


# In[21]:


match_actions = ["DeployDecoy", "Remove", "Restore", "Analyse"]
# match_actions = ["Restore"]#, "Analyse"]
agent_action_mask = {agent: [action in match_actions for action in agent_action_types[agent]] for agent in agent_names}

phase_rewards = {
   0:{
         "public_access_zone_subnet":    {"LWF": -1, "ASF": -1, "RIA": -3}, # Part of HQ Network in ReadMe
         "admin_network_subnet":         {"LWF": -1, "ASF": -1, "RIA": -3}, # Part of HQ Network in ReadMe
         "office_network_subnet":        {"LWF": -1, "ASF": -1, "RIA": -3}, # Part of HQ Network in ReadMe
         # "contractor_network_subnet":    {"LWF":  0, "ASF": -5, "RIA": -5}, # Should be 0, -5, -5, but set differently for testing
         "contractor_network_subnet":    {"LWF":  0, "ASF": 0, "RIA": 0}, # Should be 0, -5, -5, but set differently for testing
         "restricted_zone_a_subnet":     {"LWF": -1, "ASF": -3, "RIA": -1},
         "operational_zone_a_subnet":    {"LWF": -1, "ASF": -1, "RIA": -1},
         "restricted_zone_b_subnet":     {"LWF": -1, "ASF": -3, "RIA": -1},
         "operational_zone_b_subnet":    {"LWF": -1, "ASF": -1, "RIA": -1},
         "internet_subnet":              {"LWF":  0, "ASF":  0, "RIA": -1}},
   1:{
         "public_access_zone_subnet":    {"LWF": -1, "ASF": -1, "RIA": -3},
         "admin_network_subnet":         {"LWF": -1, "ASF": -1, "RIA": -3},
         "office_network_subnet":        {"LWF": -1, "ASF": -1, "RIA": -3},
         "contractor_network_subnet":    {"LWF":  0, "ASF":  0, "RIA":  0},
         "restricted_zone_a_subnet":     {"LWF": -2, "ASF": -1, "RIA": -3},
         "operational_zone_a_subnet":    {"LWF":-10, "ASF":  0, "RIA":-10},
         "restricted_zone_b_subnet":     {"LWF": -1, "ASF": -1, "RIA": -1},
         "operational_zone_b_subnet":    {"LWF": -1, "ASF": -1, "RIA": -1},
         "internet_subnet":              {"LWF":  0, "ASF":  0, "RIA": 0}},
   2:{
         "public_access_zone_subnet":    {"LWF": -1, "ASF": -1, "RIA": -3},
         "admin_network_subnet":         {"LWF": -1, "ASF": -1, "RIA": -3},
         "office_network_subnet":        {"LWF": -1, "ASF": -1, "RIA": -3},
         "contractor_network_subnet":    {"LWF":  0, "ASF":  0, "RIA":  0},
         "restricted_zone_a_subnet":     {"LWF": -1, "ASF": -3, "RIA": -3},
         "operational_zone_a_subnet":    {"LWF": -1, "ASF": -1, "RIA": -1},
         "restricted_zone_b_subnet":     {"LWF": -2, "ASF": -1, "RIA": -3},
         "operational_zone_b_subnet":    {"LWF":-10, "ASF":  0, "RIA":-10},
         "internet_subnet":              {"LWF":  0, "ASF":  0, "RIA":  0}}}

class Agent:
   def __init__(self, agent_name: str, config):
      self.agent_name = agent_name
      self.config = config
      self.reset()
      self.hosts_process_blast = set()

   def block_unblock(self, observation):
      return None 
      # mission_phase = observation[0]
      # if mission_phase == 2:
      #    keyword = "BlockTrafficZone operational_zone_b_subnet"
      #    block_actions = set(action for action in agent_action_names[self.agent_name] if keyword in action)
      #    block_actions = set(action for action in block_actions if "restricted_zone_b_subnet" in action or "contractor" in action)
      #    # block_actions = set(action for action in block_actions if "restricted_zone_b_subnet" in action)
      #    # block_actions = set(action for action in block_actions if "contractor" in action)
      #    # print(block_actions)
      #    blocks_todo = block_actions - self.blocks_done
      #    if len(blocks_todo) > 0:
      #       block_action_name = blocks_todo.pop()
      #       self.blocks_done.add(block_action_name)
      #       # print(f"{self.step}: {self.agent_name}: block {block_action_name}")
      #       return agent_action_names[self.agent_name].index(block_action_name)
      # if self.step % 2 == 0: return None
      # return None # Don't block or unblock other subnets, it's terrible
      # if self.agent_name == 'blue_agent_0': return Nonex(block_action_name)
      # if self.step % 2 == 0: return None
      # return None # Don't block or unblock other subnets, it's terrible
      # if self.agent_name == 'blue_agent_0': return None
      if self.agent_name == 'blue_agent_1': return None
      # if self.agent_name == 'blue_agent_2': return None
      if self.agent_name == 'blue_agent_3': return None
      if self.agent_name == 'blue_agent_4': return None
      subnet_action_counter = -1
      for subnet_counter in range(len(agent_subnet_idx[self.agent_name])):
         if observation[agent_subnet_idx[self.agent_name][subnet_counter]] == 1:
            continue
         else:
            subnet_action_counter += 1
         is_blocked = observation[agent_subnet_status_idx[self.agent_name][subnet_counter]] == 1
         should_block = observation[agent_subnet_should_idx[self.agent_name][subnet_counter]] == 1
         if should_block and not is_blocked:
            action_name = agent_action_names[self.agent_name][agent_block_subnet_idx[self.agent_name][subnet_action_counter]]
            # if "operational" not in action_name: return None
            # print(f"{self.step}: {self.agent_name}: block {subnet_counter} {action_name}")
            return agent_block_subnet_idx[self.agent_name][subnet_action_counter]
         elif is_blocked and not should_block:
            action_name = agent_action_names[self.agent_name][agent_allow_subnet_idx[self.agent_name][subnet_action_counter]]
            # if "operational" not in action_name: return None
            # print(f"{self.step}: {self.agent_name}: unblock {subnet_counter} {action_name}")
            return None
            return agent_allow_subnet_idx[self.agent_name][subnet_action_counter]
      return None

   def detect_malice(self, observation):
      for i, idx in enumerate(agent_malicious_process_idx[self.agent_name]):
         if observation[idx] == 1:
            self.malice.append((i, 'process'))
      for i, idx in enumerate(agent_malicious_connection_idx[self.agent_name]):
         if observation[idx] == 1:
            self.malice.append((i, 'connection'))
      # if len(self.malice)>0:
         # print(f"{self.step}: {self.agent_name}: Malice detected: {self.malice}")
      # self.interests = [i * 1.0 for i in self.interests]
      for (host,typ) in self.malice:
        self.interests[agent_restore_idx[self.agent_name][host]] += self.config.detect_increase['Restore'][typ]
        self.interests[agent_remove_idx[self.agent_name][host]] += self.config.detect_increase['Remove'][typ]
        self.interests[agent_analyse_idx[self.agent_name][host]] += self.config.detect_increase['Analyse'][typ]
        analyse_action_number = agent_analyse_idx[self.agent_name][host]
        analyse_action_name = agent_action_names[self.agent_name][analyse_action_number]
      #   self.alerts[host].append((self.step, typ))
        if IS_NOTEBOOK: print("Malice!", (host,typ), analyse_action_name)
         # IDEA!!!!!!
         # We should reset these extra weights, otherwise future observations may not have an effect.
         # Keep track of these extra priorities in a separate vector that we reset once the action is performed.

      self.malice = []
         # self.priorities[agent_restore_idx[self.agegent_name][i]] += 0.1


   def print_observation_changes(self, observation):
      return
      for i in range(len(observation)):
         if observation[i] != self.previous_observation_vector[i]:
            descr = get_observation_description(self.agent_name)[i]['descr']
            print(f"{self.step}: {self.agent_name}: {descr} went from {self.previous_observation_vector[i]} to {observation[i]}")
      self.previous_observation_vector = observation

   def is_in_subnet(self, subnet):
      return any(subnet in action and "Restore" in action for action in agent_action_names[self.agent_name])
   
   def user_transform(self, action):
      return 1
      if "user" in action:
         return self.config.user_factor[self.agent_name]
      else:
         return 1

   def get_action_prim(self, raw_observation, action_space):
      observation, cy_observation = raw_observation
      if IS_NOTEBOOK:
         if 'message' in cy_observation: del cy_observation['message']
         print(self.step, ". ", self.agent_name, ": ", cy_observation)
      self.print_observation_changes(observation)
      self.step += 1

      if "action" in cy_observation:
         act = cy_observation[ "action" ]
         if cy_observation[ "success" ]:
            if act.name == "Restore":
               self.hosts_process_blast.discard( act.hostname )

      mission_phase = self.step // ( 500 // 3 )
      bad_phase_1 = ( mission_phase == 1 ) and ( self.agent_name == "blue_agent_1" )
      bad_phase_2 = ( mission_phase == 2 ) and ( self.agent_name == "blue_agent_3" )
      bad_phase = bad_phase_1 or bad_phase_2
      
      sigs_all = {}
      lotslots = set()
      for host,v in cy_observation.items():
         if isinstance( v, dict ) and "Processes" in v.keys():
            count_con = 0
            sigs = {}
            for proc in v[ "Processes" ]:
               if "Connections" in proc.keys():
                  count_con += 1
                  for conn in proc[ "Connections" ]:
                     cks = conn.keys()
                     if "remote_port" in cks and "local_address" in cks and "remote_address" in cks:
                        sig = ( conn[ "remote_port" ], str( conn[ "local_address" ] ), str( conn[ "remote_address" ] ) )
                        if sig in sigs.keys():
                           sigs[ sig ] += 1
                        else:
                           sigs[ sig ] = 1
            if len( sigs ) > 0 and not "router" in host:
               sigs_all[ host ] = sigs
            #       if host == "public_access_zone_subnet_user_host_2":
            #       interested_action = f"Remove {host}"
            #       self.priorities[agent_action_names[self.agent_name].index(interested_action)] += 1000
            if count_con > 18:
               lotslots.add( host )
               self.hosts_process_blast.add( host )
               interested_action = f"Analyse {host}"
               self.priorities[agent_action_names[self.agent_name].index(interested_action)] += self.config.connections_increase

      sus_remove = set()
      for host_a, sigs_a in sigs_all.items():
         for sig_a, count_a in sigs_a.items():
            port_a = sig_a[ 0 ]
            if count_a > 1 and port_a == 22:
               for host_b, sigs_b in sigs_all.items():
                  if host_a <= host_b:
                     continue
                  for sig_b, count_b in sigs_b.items():
                     if sig_a[ 1 ] == sig_b[ 1 ] and sig_a[ 2 ] == sig_b[ 2 ]:
                        if host_a in lotslots:
                           sus_remove.add( host_b )
                        if host_b in lotslots:
                           sus_remove.add( host_a )

      for host in sus_remove:
         interested_action = f"Restore {host}"
         self.priorities[agent_action_names[self.agent_name].index(interested_action)] += self.config.priority_decrease[ "Restore" ]

      self.detect_malice(observation)
      if self.progress_todo > 0:
         if not( "PROGRESS" in str(cy_observation) ):
            print( f"{self.step} PROGRESS FAIL {self.agent_name} {cy_observation}" )
            assert False
         self.progress_todo -= 1
         return action_type_masks[self.agent_name]['Monitor'].index(True)
      self.active_step += 1
      action_mask = [x for x in action_space['mask']]
      action_mask = [action_mask[i] and agent_action_mask[self.agent_name][i] for i in range(len(action_mask))]
      indices = [i for i, x in enumerate(action_mask) if x]
      # check if analyse gave a result
      str_obs = str(cy_observation)
      if "Analyse" in str_obs and not "escalate.sh" in str_obs and not "cmd.sh" in str_obs:
         action, host = str(cy_observation['action']).split(' ')
         interested_action = f"Analyse {host}"
         self.interests[agent_action_names[self.agent_name].index(interested_action)] -= self.config.analyse_penalties['None']*self.user_transform(interested_action)
      if "escalate.sh" in str_obs or "cmd.sh" in str_obs:
         action, host = str(cy_observation['action']).split(' ')
         assert action == "Analyse"
         if "escalate.sh" in str_obs:
            next_action = "Restore"
            self.num_restores[host] += 1
            # increase Decoy priority on this host by 1000
            if any("operational_zone_a_subnet_user_host" in action for action in agent_action_names[self.agent_name]):
               maxstep = 500/3
            else:
               maxstep = 2*500/3
            if self.step < maxstep and "user" in host:# and self.agent_name != 'blue_agent_4':
               interested_action = f"DeployDecoy {host}"
               self.priorities[agent_action_names[self.agent_name].index(interested_action)] += self.config.restore_decoy_increase
            interested_action = f"Analyse {host}"
            self.interests[agent_action_names[self.agent_name].index(interested_action)] -= self.config.analyse_penalties['Restore']*self.user_transform(interested_action)
         else:
            interested_action = f"Analyse {host}"
            self.interests[agent_action_names[self.agent_name].index(interested_action)] -= self.config.analyse_penalties['Remove']*self.user_transform(interested_action)
            if bad_phase and ( host in self.hosts_process_blast ) and False:
               next_action = "Restore"
            else:
               next_action = "Remove"
            # if random.random() < 0.1:
               # next_action = "Restore"
            # increase Decoy priority on this host by 1000
            # interested_action = f"DeployDecoy {host}"
            # self.priorities[agent_action_names[self.agent_name].index(interested_action)] += 1000
            # interested_action = f"Analyse {host}"
            # self.interests[agent_action_names[self.agent_name].index(interested_action)] -= 1
         # decrease Analyse priority on this host by 1
         next_action = next_action + " " + host
         next_action_number = agent_action_names[self.agent_name].index(next_action)
         choice = next_action_number
      else: 
         # # # check if we need to block or unblock a subnet
         # block_action = self.block_unblock(observation)
         # if block_action is not None:
         #    self.progress_todo = agent_action_durations[self.agent_name][block_action] - 1
         #    return block_action
         choice = max(indices, key=lambda action: (self.priorities[action] + self.interests[action]))
         action_name = agent_action_names[self.agent_name][choice]
         penalty_factor = 1
         if "Analyse" in action_name:
            host = action_name.split(' ')[1]
            if self.num_restores[host] > 0:
               penalty_factor *= self.config.restore_priority_factor
         if 'user' in agent_action_names[self.agent_name][choice]:
            self.priorities[choice] -= self.config.priority_decrease[agent_action_types[self.agent_name][choice]]*self.config.user_factor[self.agent_name]*penalty_factor
         else:
            self.priorities[choice] -= self.config.priority_decrease[agent_action_types[self.agent_name][choice]]*penalty_factor
      # check if it's a decoy action
      if "DeployDecoy" in agent_action_names[self.agent_name][choice]:
         self.decoy_count += 1
      self.progress_todo = agent_action_durations[self.agent_name][choice] - 1
      return choice
   
   def get_action(self, observation, action_space):
      choice = self.get_action_prim(observation, action_space)
      # print the action name if it's a restore action
      if IS_NOTEBOOK:
         print(f"{agent_action_names[self.agent_name][choice]} ({self.step})")
      return choice

   def reset(self):
      self.previous_observation_vector = [-1 for _ in range(agent_observation_lengths[self.agent_name])]
      self.progress_todo = 0
      self.step = 0
      self.active_step = 0
      self.priorities = [
         self.config.initial_priorities.get(agent_action_types[self.agent_name][action], 0.0)
         for action in range(action_space_sizes[self.agent_name])
      ]
      # decrease all decoy priorities on server hosts by 1000
      for i in range(len(self.priorities)):
         if ("DeployDecoy" in agent_action_names[self.agent_name][i] and "server" in agent_action_names[self.agent_name][i]):# or self.agent_name == 'blue_agent_4':
            self.priorities[i] -= 1000
      self.interests = [
         self.config.initial_interests.get(agent_action_types[self.agent_name][action], 0.0)
         for action in range(action_space_sizes[self.agent_name])
      ]
      self.malice = []
      self.blocks_done = set()
      self.did_increase = False
      self.decoy_count = 0
      self.hosts = [a.split(' ')[1] for a in agent_action_names[self.agent_name] if "DeployDecoy" in a]
      self.num_restores = {host: 0 for host in self.hosts}
      # self.last_clean = {host: 0 for host in self.hosts}
      # self.alerts = {host: [] for host in self.hosts}


class MaskWrapper:
    def __init__(self, env, agents, seed = None):
        self.env = env
        self.mae = EnterpriseMAE(env)
        self.agents = agents
        if IS_NOTEBOOK:
           self.logfile = open("log.html", "w")

    def reset(self, seed=None):
        self.step_count = -1
        for agent in self.agents.values():
            agent.reset()
        obs, rest = self.mae.reset( seed=seed )
        return self.add_cyborg_observations(obs), rest

    def action_space(self, agent_name):
        return self.mae.get_action_space(agent_name)

    def set_seed(self,seed=None):
       self.env.set_seed( seed )

    def get_last_action(self, agent: str):
        return self.env.get_last_action(agent=agent)

    def step(self, actions):
      self.step_count += 1
      observations, rew, term, trunc, info = self.mae.step(actions)
      return self.add_cyborg_observations(observations), rew, term, trunc, info

    def add_cyborg_observations(self, observations):
       raw_observations = {}
       for agent_name in self.agents:
          raw_observations[agent_name] = (observations[agent_name], self.env.get_observation(agent_name))
       return raw_observations


# In[22]:


class Logger:
  def __init__(self, state):
    self.outfile = open("log2.html", "w")
    self.initial_state = state

  def write(self, x):
    self.outfile.write(x)
    self.outfile.write("\n")

  def log(self, state):
    return
    self.write("<!DOCTYPE html>")
    self.write("""<html><head>""")
    self.write("""<link rel="stylesheet" href="style.css">""")
    self.write("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">""")
    self.write("""</head><body><br><br><br><br>""")

    hosts = sorted(x for x in state.keys() if x != 'success')
    diff_state = {host: {k: [x for x in v if x not in self.initial_state[host].get(k, [])] for k, v in state[host].items()} for host in hosts}

    def tooltip(content, tooltip):
      return f"""<span class="tooltip">{content}<span class="tooltiptext">{tooltip}</span></span>"""

    def icontooltip(icon, text, tt):
      icon = f"""<i class="fas fa-{icon}"></i>"""
      return f"""<span class="tooltip">{tooltip(icon, tt)}</span>"""

    self.write("<table>")
    self.write("<tr>")
    for hostname in hosts:
      short = hostname.replace("_", "").replace("router", "R").replace("user", "U").replace("server", "S")
      short = short.replace("network", "").replace("admin", "A<br>lambda host: self.last_clean_fn(host) - self.host_factor(host)t", "I").replace("subnet", "").replace("zone", "")
      short = short.replace("host", "").replace("operationala", "a<br>").replace("operationalb", "b<br>")
      short = short.replace("restricteda", "A<br>").replace("restrictedb", "B<br>")
      short = short.replace("publicaccess", "P<br>").replace("root", "")
      self.write(f"""<td>{tooltip(short, hostname)}</td>""")
    self.write("</tr>")
    self.write("<tr>")
    for hostname in hosts:
      self.write(f"""<td>""")
      host = diff_state[hostname]
      # self.write(f"""<span class="host">{tooltip(hostname, host)}</span>""")
      # self.write("<div>")
      for process in host.get('Processes', []):
        self.write(f"""<span class="process">{icontooltip("cog", process.get('process_name',"???"),disp(process))}</span>""")
      for file in host.get('Files', []):
        self.write(f"""<span class="file">{icontooltip("file", file['File Name'],disp(file))}</span>""")
      for session in host.get('Sessions', []):
        self.write(f"""<span class="session">{icontooltip("wifi",session['Type'],disp(session))}</span>""")
      # self.write("</div>")
      self.write("</td>")
    self.write("</body></html>")

  def __del__(self):
    self.outfile.close()


# In[23]:


configs = []
def rnd(x=0.0, y=1.0):
   # return 0
   return random.random() * (y-x) + x
N = 100000 if not IS_NOTEBOOK else 2
for i in range(N):
   config = {
      'steps': 500,
      # 'initial_priorities': {
      #    'Restore': 0.0+rnd(),
      #    'DeployDecoy': 1.0+rnd(),
      #    'Remove': 0.0+rnd(),
      # },
      # 'priority_decrease': {
      #    'Restore': 1.5+rnd(),
      #    'DeployDecoy': 0.75,#+rnd(),
      #    'Remove': 1.0+rnd(),
      # },

      'connections_increase': 0.0,
      'initial_priorities': {
         'Restore': 0.0,
         'DeployDecoy': 1000.0, #1000.0, #0.001,
         'Remove': 0.0, #0.01,
         'Analyse': 1000.0
      },
      'priority_decrease': {
         'Restore': 1000.0, #.5, #1.5, #2.0,#1.5+rnd(),
         'DeployDecoy': 1000.0, #0.75,#+rnd(),
        #  'Remove': 0.5,
         'Remove': 1000.0,
         'Analyse': 2.375,
         # 'Analyse': 3.818,
      },
      # 'priority_decrease': {
      #    'Restore': 5, #1.5, #2.0,#1.5+rnd(),
      #    'DeployDecoy': 2, #0.75,#+rnd(),
      #    'Remove': 3,
      #    'Analyse': 2.0,
      # },
      'initial_interests': {
         'Restore': 0.0,
         'DeployDecoy': 0.001,
         'Remove': 0.001,
         'Analyse': 0.0
      },
      'detect_increase': {
         'Restore': 0.0005,
         # 'Analyse': 0.0005,
         # 'Analyse': {'process': rnd(0.0, 2.0), 'connection': 0.5},
         'Analyse': {'process': 1.9, 'connection': 0.25},
         # 'Analyse': {'process': 0.05, 'connection': 0.5},
         'Remove': {'process': 0.0, 'connection': 0.0},
         'Restore': {'process': 0.0, 'connection': 0.0},
      },
      'user_factor': {
         'blue_agent_0': 0.6,
         'blue_agent_1': 0.6,
         'blue_agent_2': 0.6,
         'blue_agent_3': 0.6,
         'blue_agent_4': 0.6,
      },
      'cmd_increase': 0.0000,
      'round_robin': False,
      'restore_priority_factor': 1,
      'restore_decoy_increase': 1000.0,
      'analyse_penalties': {
         'Restore': 0.51234,
         'Remove': 0, #-0.15, #rnd(-0.5, 0.5),
         'None': 0.5,
      },
      'reverse_indices': 0 #random.choice([1, 0])
   }
   configs.append(config)
# random.shuffle(configs)


# In[24]:


if __name__ == "__main__":
   import wandb
   W = wandb.init(project="cyborg-manual2", name="decay") if not IS_NOTEBOOK else None
   n = 0
   s = 0
   for input, result in run_par(run_sim, configs):
      n += 1
      s += result
      if W:
         wandb.log({"C": input, "reward": result, "mean_reward": s/n, "n": n})
      print(input, f"\n {n} reward: ", result)
   # exit() if not ISss_NOTEBOOK else None


# In[ ]:





# In[ ]:




