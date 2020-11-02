from tensorforce.environments.environment import Environment
import numpy as np
import math
import signal
import time

import logging

# Load UnityEnvironment and my wrapper
from mlagents.envs import UnityEnvironment

eps = 1e-10


class UnityEnvWrapper(Environment):
    def __init__(self, game_name = None, no_graphics = True, seed = None, worker_id=0, size_global = 8, size_two = 5, with_local = True, size_three = 3, with_stats=True, size_stats = 1,
                 with_previous=True, manual_input = False, config = None, curriculum = None, verbose = False, agent_separate = False, agent_stats = 6,
                 with_class=False, with_hp = False, size_class = 3, double_agent = False, embedding_type='dense_embedding'):

        self.probabilities = []
        self.size_global = size_global
        self.size_two = size_two
        self.with_local = with_local
        self.size_three = size_three
        self.with_stats = with_stats
        self.size_stats = size_stats
        self.manual_input = manual_input
        self.with_previous = with_previous
        self.config = config
        self.curriculum = curriculum
        self.verbose = verbose
        self.agent_separate = agent_separate
        self.agent_stats = agent_stats
        self.with_class = with_class
        self.with_hp = with_hp
        self.size_class = size_class
        self.double_agent = double_agent
        self.game_name = game_name
        self.no_graphics = no_graphics
        self.seed = seed
        self.worker_id = worker_id
        self.unity_env = self.open_unity_environment(game_name, no_graphics, seed, worker_id)
        self.default_brain = self.unity_env.brain_names[0]
        self.input_channels = 6
        super(UnityEnvWrapper, self).__init__()
        self._max_episode_timesteps = 100

        self.one_hot = False

        if embedding_type == 'dense_embedding':
            self.get_input_observation = self.get_input_observation_dense
            self.print_observation = self.print_observation_dense
        elif embedding_type == 'transformer':
            self.get_input_observation = self.get_input_observation_transformer
            self.print_observation = self.print_observation_transformer
            self.with_transformer = True

    count = 0

    def to_one_hot(self, a, channels):
        return (np.arange(channels) == a[..., None]).astype(float)

    # This method change the vetorial state output by the game and transform it in
    # state_spec form for the Transformer
    def get_input_observation_transformer(self, env_info, action = None):
        global_size = self.size_global * self.size_global

        # Get the global view of cell type
        global_in = env_info.vector_observations[0][:global_size]
        global_in = np.reshape(global_in, (self.size_global, self.size_global))

        # Get the local view of cell type
        local_size = self.size_two * self.size_two
        local_in = env_info.vector_observations[0][global_size:(global_size + (local_size))]
        local_in = np.reshape(local_in, (self.size_two, self.size_two))

        # Get the local_two view of cell type
        local_two_size = self.size_three * self.size_three
        local_in_two = env_info.vector_observations[0][(global_size + (local_size)):(
                global_size + (local_size) + (local_two_size))]
        local_in_two = np.reshape(local_in_two, (self.size_three, self.size_three))

        stats = env_info.vector_observations[0][
                (global_size + (local_size) + (local_two_size)):
                (global_size + local_size + local_two_size + self.size_stats)]

        agent_spec = (1, 0)
        agent_size = agent_spec[0] * agent_spec[1]
        target_spec = (1, 0)
        target_size = target_spec[0] * target_spec[1]
        
        current_index = global_size + local_size + local_two_size + self.size_stats

        # Get the melee loot spec
        melee_spec = (20,44)
        melee_size = melee_spec[0] * melee_spec[1]
        melee_weapons = env_info.vector_observations[0][
                (current_index +
                 agent_size + target_size):
                (current_index +
                 agent_size + target_size + melee_size)]
        melee_weapons = np.reshape(melee_weapons, melee_spec)

        # Get the ranged loot spec
        range_spec = (20, 44)
        range_size = range_spec[0] * range_spec[1]
        range_weapons = env_info.vector_observations[0][
                      (current_index +
                       agent_size + target_size +melee_size):
                      (current_index +
                       agent_size + target_size +melee_size +
                       range_size)]
        range_weapons = np.reshape(range_weapons, range_spec)

        # Get the potions loot spec
        potions_spec = (20, 44)
        potions_size = potions_spec[0] * potions_spec[1]
        potions = env_info.vector_observations[0][
                          (current_index +
                           agent_size + target_size + melee_size +
                           range_size):
                          (current_index +
                           agent_size + target_size + melee_size +
                           range_size + potions_size)]
        potions = np.reshape(potions, potions_spec)

        # Get the global and relative position of loot objects
        total_entities = melee_spec[0] + range_spec[0] + potions_spec[0]
        global_positions = env_info.vector_observations[0][(current_index +
                           agent_size + target_size + melee_size +
                           range_size + potions_size): (current_index +
                           agent_size + target_size + melee_size +
                           range_size + potions_size) + total_entities]

        local_positions = env_info.vector_observations[0][(current_index +
                           agent_size + target_size + melee_size +
                           range_size + potions_size) + total_entities: (current_index +
                           agent_size + target_size + melee_size +
                           range_size + potions_size) + total_entities * 2]

        local_two_positions = env_info.vector_observations[0][(current_index +
                           agent_size + target_size + melee_size +
                           range_size + potions_size) + total_entities * 2:]

        action_vector = np.zeros(17)
        if action != None:
            action_vector[action] = 1

        agent_stats_size = 13
        # Create the proper multi state dict to be fed to net
        observation = {
            'global_in': global_in,
            'local_in': local_in,
            'local_in_two': local_in_two,
            'melee_weapons': melee_weapons,
            'range_weapons': range_weapons,
            'potions': potions,

            'global_indices': global_positions,
            'local_indices': local_positions,
            'local_two_indices': local_two_positions,

            'agent_stats': stats[:agent_stats_size],
            'target_stats': stats[agent_stats_size:],
            'prev_action': action_vector
        }

        return observation

    # This method change the vetorial state output by the game and transform it in
    # state_spec form for the DenseEmbedding
    def get_input_observation_dense(self, env_info, action=None):
        size = self.size_global * self.size_global * self.input_channels

        # Get the global view of map and objects
        global_in = env_info.vector_observations[0][:size]
        global_in = np.reshape(global_in, (self.size_global, self.size_global, self.input_channels))
        # Transform the categorical multi-channel map into one-hot multi-channel map
        global_in_one_hot = self.to_one_hot(global_in[:, :, 0], 7)
        for i in range(1, self.input_channels):
            global_in_one_hot = np.append(global_in_one_hot, self.to_one_hot(global_in[:, :, i], 9), axis=2)

        # Get the local view of map and objects
        local_size = self.size_two * self.size_two
        local_size = local_size * self.input_channels
        local_in = env_info.vector_observations[0][size:(size + local_size)]
        local_in = np.reshape(local_in, (self.size_two, self.size_two, self.input_channels))
        # Transform the categorical multi-channel map into one-hot multi-channel map
        local_in_one_hot = self.to_one_hot(local_in[:, :, 0], 7)
        for i in range(1, self.input_channels):
            local_in_one_hot = np.append(local_in_one_hot, self.to_one_hot(local_in[:, :, i], 9), axis=2)

        # Get the local_two view of map and objects
        local_two_size = self.size_two * self.size_two
        local_two_size = local_two_size * self.input_channels
        local_in_two = env_info.vector_observations[0][(size + local_size):(
                size + local_size + local_two_size)]
        local_in_two = np.reshape(local_in_two, (self.size_three, self.size_three, self.input_channels))
        # Transform the categorical multi-channel map into one-hot multi-channel map
        local_in_two_one_hot = self.to_one_hot(local_in_two[:, :, 0], 7)
        for i in range(1, self.input_channels):
            local_in_two_one_hot = np.append(local_in_two_one_hot, self.to_one_hot(local_in_two[:, :, i], 9),
                                             axis=2)

        # Get the agent and enemy stats
        stats = env_info.vector_observations[0][
                (size + local_size + local_two_size):
                (size + local_size + local_two_size + self.size_stats)]

        agent_stats_size = 13
        agent_stats = stats[:agent_stats_size]
        target_stats = stats[agent_stats_size:]

        action_vector = np.zeros(17)
        action_vector[action] = 1

        observation = {
            'global_in': global_in_one_hot,

            'local_in': local_in_one_hot,

            'local_in_two': local_in_two_one_hot,

            'agent_stats': agent_stats,
            'target_stats': target_stats,
            'prev_action': action_vector
        }

        return observation

    def execute(self, actions):

        assert self.timestep < self._max_episode_timesteps

        if self.manual_input:
            input_action = input('...')

            try:
                actions = int(input_action)
            except ValueError:
                pass

        env_info = None
        signal.alarm(0)
        while env_info == None:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(3000000000000000000)
            try:
                env_info = self.unity_env.step([actions])[self.default_brain]
            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed = int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        if self.double_agent:
            while len(env_info.vector_observations) <= 0:
                env_info = self.unity_env.step()[self.default_brain]

        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        observation = self.get_input_observation(env_info, actions)

        self.count += 1

        if self.verbose:
            print('action = ' + str(actions))
            print('reward = ' + str(reward))
            print('timestep = ' + str(self.count))
            self.print_observation(observation)

        self.timestep += 1
        if int(done) == 0 and self.timestep >= self._max_episode_timesteps:
            done = 2

        return [observation, done, reward]

    def set_config(self, config):
        self.config = config

    def handler(self, signum, frame):
        print("Timeout!")
        raise Exception("end of time")

    def reset(self):

        self.count = 0
        self.timestep = 0
        env_info = None

        while env_info == None:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(10000000000000)
            try:
                logging.getLogger("mlagents.envs").setLevel(logging.WARNING)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed=int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        if self.double_agent:
            while len(env_info.vector_observations) <= 0:
                env_info = self.unity_env.step()[self.default_brain]

        observation = self.get_input_observation(env_info)

        if self.verbose:
            self.print_observation(observation)

        return observation

    def print_observation_transformer(self, observation):
        try:
            print(observation)
            print(observation['global_in'])
            print(observation['local_in'])
            print(observation['local_in_two'])

            print('')
            print(observation['agent_stats'])
            print(observation['target_stats'])

            print('')
            print(observation['melee_weapons'])
            print(observation['range_weapons'])
            print(observation['potions'])

        except Exception as e:
            pass

    def print_observation_dense(self, observation):
        try:
            sum = observation['global_in'][:, :, 0] * 0
            for i in range(1, self.input_channels + 1):
                sum += observation['global_in'][:, :, i] * i
            sum = np.flip(np.transpose(sum), 0)
            print(sum)
            print(' ')
            print(observation['agent_stats'])
            print(observation['target_stats'])

        except Exception as e:
            pass

    def close(self):
        self.unity_env.close()

    def open_unity_environment(self, game_name, no_graphics, seed, worker_id):
        return UnityEnvironment(game_name, no_graphics=no_graphics, seed=seed, worker_id=worker_id)

    def add_probs(self, probs):
        self.probabilities.append(probs[0])

    def get_last_entropy(self):
        entropy = 0
        for prob in self.probabilities[-1]:
            entropy += (prob + eps)*(math.log(prob + eps) + eps)

        return -entropy

    def states(self):
        return self.states_spec

    def actions(self):
        return {
                    'type': 'int',
                    'num_values': 17
                }

    def max_episode_timesteps(self):
        return 100

    def set_states(self, states_spec):
        self.states_spec = states_spec


class Info():
    def __init__(self, string):
        self.item = string

    def items(self):

        return self.item, self.item
