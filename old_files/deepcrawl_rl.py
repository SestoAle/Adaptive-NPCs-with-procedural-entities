import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from reward_model.reward_model_tf import RewardModel
import tensorflow as tf
# Check if CUDA is available
#use_cuda = tf.test.is_gpu_available()
use_cuda = True
#use_cude = tf.config.list_physical_devices('GPU') 
from tensorforce.agents import PPOAgent, VPGAgent, DQNAgent
from tensorforce.execution import Runner
from reward_model.utils import NumpyEncoder

import time
import os

import json


from unity_env_wrapper import UnityEnvWrapper
from export_graph import export_pb
from deepcrawl_runner import DeepCrawlRunner

import argparse

import datetime

if use_cuda:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    execution = dict(
        type='single',
        session_config = tf.ConfigProto(gpu_options=gpu_options),
        distributed_spec=None
    )
else:
    gpu_options = None
    execution = dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )

'''--------------------------'''
'''         Arguments        '''
'''--------------------------'''

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model-name', help="The name of the model", default="")
parser.add_argument('-gn', '--game-name', help="The name of the environment", default=None)
# TODO: delete this
#parser.add_argument('-gn', '--game-name', help="The name of the environment", default=None)
parser.add_argument('-ne', '--num-episodes', help="Specify the number of episodes after which the environment is restarted", default=None)
parser.add_argument('-wk', '--worker-id', help="The id for the worker", default=0)
parser.add_argument('-mt', '--num-timesteps', help="Max timesteps of the agent", default=100)
parser.add_argument('-ls', '--lstm', help="If the net has an lstm", default=True)
parser.add_argument('-un', '--update-number', help="Update dict", default=10)
parser.add_argument('-ws', '--with-stats', help="Update dict", default=True)
parser.add_argument('-sa', '--save', help="Number of episodes for saving models", default=3000)

# TODO: Cambiare
# IRL Settings
parser.add_argument('-np', '--num-policy-update', help="IRL", default=10)
parser.add_argument('-nd', '--num-discriminator-exp', help="IRL", default=10)
parser.add_argument('-rm','--reward-model', help="IRL", default=None)
parser.add_argument('-gd','--get-demonstrations', dest='get_demonstrations', action='store_true')
parser.add_argument('-fr','--fixed-reward-model', dest='fixed_reward_model', action='store_true')
parser.add_argument('-tr','--try-reward-model', dest='try_reward_model', action='store_true')
parser.add_argument('-dn','--demonstrations-name', default=None)
parser.set_defaults(get_demonstrations=False)
parser.set_defaults(fixed_reward_model=False)
parser.set_defaults(try_reward_model=False)

args = parser.parse_args()


'''--------------------------'''
'''   Algorithm parameters   '''
'''--------------------------'''

# Number of timesteps within an episode
num_timesteps = args.num_timesteps

from net_structures.net_structures import dc2_net_conv_same_stat as net
from net_structures.net_structures import dc2_baseline_same_stat as baseline

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    # Inputs structure
    states={
        # Global View
        'global_in': {'shape': (10, 10, 52), 'type': 'float'},
        # 'attr_global_1': {'shape': (10, 10), 'type': 'int'},
        # 'attr_global_2': {'shape': (10, 10), 'type': 'int'},
        # 'attr_global_3': {'shape': (10, 10), 'type': 'int'},
        # 'attr_global_4': {'shape': (10, 10), 'type': 'int'},
        # 'attr_global_5': {'shape': (10, 10), 'type': 'int'},
        #'global_in_attributes': {'shape': (10, 10, 5), 'type': 'float'},

        # Local View
        'local_in': {'shape': (5, 5, 52), 'type': 'float'},
        # 'attr_local_1': {'shape': (5, 5), 'type': 'int'},
        # 'attr_local_2': {'shape': (5, 5), 'type': 'int'},
        # 'attr_local_3': {'shape': (5, 5), 'type': 'int'},
        # 'attr_local_4': {'shape': (5, 5), 'type': 'int'},
        # 'attr_local_5': {'shape': (5, 5), 'type': 'int'},
        #'local_in_attributes': {'shape': (5, 5, 5), 'type': 'float'},

        # Local 2 View
        'local_in_two': {'shape': (3, 3, 52), 'type': 'float'},
        # 'attr_local_two_1': {'shape': (3, 3), 'type': 'int'},
        # 'attr_local_two_2': {'shape': (3, 3), 'type': 'int'},
        # 'attr_local_two_3': {'shape': (3, 3), 'type': 'int'},
        # 'attr_local_two_4': {'shape': (3, 3), 'type': 'int'},
        # 'attr_local_two_5': {'shape': (3, 3), 'type': 'int'},
        #'local_in_two_attributes': {'shape': (3, 3, 5), 'type': 'float'},

        # Stats
        'stats': {'shape': (11), 'type': 'int'},
        #'target_stats': {'shape': (3), 'type': 'int'},
        'action': {'shape': (17), 'type': 'float'}
    },
    # Actions structure
    actions={
        'type': 'int',
        'num_actions': 17
    },
    network=net,
    # Agent
    states_preprocessing=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        # 10 episodes per update
        batch_size=int(args.update_number),
        # Every 10 episodes
        frequency=int(args.update_number) - 5
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=int(args.update_number) * (num_timesteps)
    ),
    # DistributionModel
    distributions=None,

    discount = 0.99,
    entropy_regularization=0.01,
    gae_lambda = None,
    likelihood_ratio_clipping=0.2,

    baseline_mode='states',
    baseline = dict(
        type = 'custom',
        network = baseline
    ),
    baseline_optimizer = dict(
        type = 'multi_step',
        optimizer = dict(
            type = 'adam',
            learning_rate = 5e-4
        ),
        num_steps = 5
    ),

    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate= 5e-6
    ),
    subsampling_fraction=0.33,
    optimization_steps=20,
    execution=execution
)

# Work ID of the environment. To use the unity editor, the ID must be 0. To use more environments in parallel, use
# different ids
work_id = int(args.worker_id)

# Number of episodes of a single run
num_episodes = args.num_episodes
# True if there is an LSTM layer
lstm = args.lstm

# Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
curriculum = {
    'current_step': 0,
    'thresholds': [3.5e6, 3e6, 2.5e6, 2e6],
    'parameters':
        {
            'minTargetHp': [1,10,10,10,15],
            'minAgentHp': [5,5,5,5,5],
            'numActions': [17,17,17,17,17],
            'maxTargetHp': [1,10,20,20,20],
            'maxAgentHp': [20,20,20,20,20],
            'maxNumLoot': [0.2,0.2,0.2,0.2,0.2],
            'minNumLoot': [0.2,0.2,0.2,0.08,0.04],
            # Agent statistics
            'agentAtk': [4, 4, 4, 4, 4],
            'agentDef': [3, 3, 3, 3, 3],
            'agentDes': [0, 0, 0, 0, 0]
        }
}

model_name = args.model_name

# Name of the enviroment to load. To use Unity Editor, must be None
game_name = args.game_name

'''--------------------------'''
'''     Run the algorithm    '''
'''--------------------------'''

use_model = None

if model_name == "" or model_name == " " or model_name == None:

    while(use_model != 'y' and use_model != 'n'):
        use_model = input('Do you want to use a previous saved model? [y/n] ')

    if (use_model == 'y'):
        model_name = input('Name of the model: ')
    else:
        model_name = input('Specify the name to save the model: ')
try:
    with open("arrays/" + model_name + ".json") as f:
        history = json.load(f)
except:
    history = None

print('')
print('--------------')
print('Agent stats: ')
print('Optimizer: ' + str(agent.optimizer['optimizer']['optimizer']))
print('Baseline: ' + str(agent.baseline_optimizer['optimizer']))
print('Discount: ' + str(agent.discount))
print('Update mode: ' + str(agent.update_mode))
print('Work id ' + str(work_id))
print("Game name: " + str(game_name))
print('--------------')
print('Net config: ' + str(agent.network))
print('--------------')
print('')

step = 0
reward = 0.0
ist_step = 0
start_time = time.time()

environment = None

# Callback function printing episode statistics
def episode_finished(r, worker_id, num_callback_episodes = 100):
    global step
    global reward
    global ist_step
    global start_time
    step += 1
    ist_step += r.episode_timestep
    reward += r.episode_rewards[-1]
    #print('Reward @ episode {}: {}'.format(step, np.mean(r.real_episode_rewards[-1:])))
    if((step % num_callback_episodes) == 0):
        print('Average cumulative estimated reward for ' + str(num_callback_episodes) + ' episodes @ episode ' + str(step) + ': ' + str(np.mean(r.episode_rewards[-num_callback_episodes:])))
        print('Average cumulative real reward for ' + str(num_callback_episodes) + ' episodes @ episode ' + str(step) + ': ' + str(np.mean(r.real_episode_rewards[-num_callback_episodes:])))
        if r.reward_model is not None and not args.fixed_reward_model:
            print('Reward Model Loss @ episode {}: {}'.format(step, np.mean(r.reward_model_loss[-num_callback_episodes:])))
            print('Reward Model Validation Loss @ episode {}: {}'.format(step, np.mean(r.reward_model_val_loss[-num_callback_episodes:])))
        print('The agent made ' + str(sum(r.episode_timesteps)) + ' steps so far')
        timer(start_time, time.time())
        reward = 0.0

    # If num_episodes is not defined, save the model every 3000 episodes
    if(num_episodes == None):
        save = int(args.save)
    else:
        save = num_episodes

    if(step % save == 0):
       save_model(r, r.reward_model)

    return True

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def save_model(runner, reward_model = None):
    global history
    # Save the runner statistics
    history = {
        "episode_rewards": runner.episode_rewards,
        "real_episode_rewards": runner.real_episode_rewards,
        "episode_timesteps": runner.episode_timesteps,
        "mean_entropies": runner.mean_entropies,
        "std_entropies": runner.std_entropies,
        "reward_model_loss": runner.reward_model_loss,
    }

    # Save the model
    runner.agent.save_model('saved/' + model_name, append_timestep=False)

    # If IRL, save reward model
    if reward_model is not None and not args.fixed_reward_model:
        reward_model.save_model(model_name + "_" + str(len(runner.episode_timesteps)))

    # Save runner statistics
    json_str = json.dumps(history, cls=NumpyEncoder)
    f = open("arrays/" + model_name + ".json", "w")
    f.write(json_str)
    f.close()

def load_model(agent):
    directory = os.path.join(os.getcwd(), "saved/")
    agent.restore_model(directory, model_name)

'''--------------------------'''
''' Initialize Reward Model  '''
'''--------------------------'''

GAN_reward = None
# sess = tf.Session()
# GAN_reward = RewardModel(obs_size=12, inner_size=10, actions_size=17, policy=agent, sess=sess)
# init = tf.global_variables_initializer()
# sess.run(init)


try:
    while True:

        if game_name == None:
            print("You're starting the training with Unity Editor. You can test the correct interactions between "
                  "Unity and Tensorforce, but for a complete training you must start it with a built environment.")

        # Close the environment
        if environment != None:
            environment.close()

        # If model name is not None, restore the parameters
        if use_model == 'y':
            directory = os.path.join(os.getcwd(), "saved/")
            agent.restore_model(directory, model_name)

        # Open the environment with all the desired flags
        environment = UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time()),
                                      worker_id=work_id, with_stats=args.with_stats, size_stats=11,
                                      size_global=10, agent_separate=False, with_class=False, with_hp=False,
                                      with_previous=lstm, verbose=False, manual_input=False)

        '''--------------------------'''
        '''       Reward Model       '''
        '''--------------------------'''

        # If model name is not None, restore the parameters of IRL model
        if use_model == 'y' and GAN_reward is not None and args.reward_model is None:
            GAN_reward.load_model(model_name)
            print("Model loaded!")

        # If use_cuda is true, move the model to GPU
        #if use_cuda and GAN_reward is not None:
           # GAN_reward = GAN_reward.cuda()

        # TODO: CAMBIARE
        if args.reward_model is not None and GAN_reward is not None and args.firxed_reward:
            GAN_reward.load_model(args.reward_model)
            print("Model loaded!")


        if args.try_reward_model:
            GAN_reward.create_demonstrations(environment, False, True)
            break

        # Create the runner to run the algorithm
        runner = DeepCrawlRunner(agent=agent, environment=environment, history=history, curriculum=curriculum,
                                 reward_model=GAN_reward, model_name=model_name,
                                 num_discriminator_exp=int(args.num_discriminator_exp),
                                 num_policy_updates=int(args.num_policy_update),
                                 dems_name = args.demonstrations_name)

        # Start learning for num_episodes episodes. After that, save the model, close the environment and reopen it.
        # Do this to avoid memory leaks or environment errors
        runner.run(episodes=num_episodes, max_episode_timesteps=num_timesteps, episode_finished=episode_finished, fixed_reward_model=args.fixed_reward_model)

        use_model = 'y'

        print('')
        print('--------------')
        print('Agent stats: ')
        print('Optimizer: ' + str(agent.optimizer['optimizer']['optimizer']))
        print('Baseline: ' + str(agent.baseline_optimizer['optimizer']))
        print('Discount: ' + str(agent.discount))
        print('Update mode: ' + str(agent.update_mode))
        print('Work id ' + str(work_id))
        print("Game name: " + str(game_name))
        print('--------------')
        print('Net config: ' + str(agent.network))
        print('--------------')
        print('')

finally:
    
    '''--------------------------'''
    '''      End of the run      '''
    '''--------------------------'''

    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=runner.episode,
        ar=np.mean(runner.episode_rewards[-100:]))
    )

    '''--------------------------'''
    '''     Try reward model     '''
    '''--------------------------'''

    # Save the model and the runner statistics
    if model_name == "" or model_name == " " or model_name == None:
        saveName = input('Do you want to specify a save name? [y/n] ')
        if(saveName == 'y'):
            saveName = input('Specify the name ')
        else:
            saveName = 'Model'
    else:
        saveName = model_name

    save_model(runner, GAN_reward)

    # Close the runner
    runner.close()

    # Freeze the TensorFlow graph and save .bytes file. All the output layers to fetch must be specified
    if lstm:
        export_pb('./saved/' + saveName, 'ppo/actions-and-internals/categorical/sample/Select,ppo/actions-and-internals/layered-network/apply/internal_lstm0/apply/stack')
    else:
        export_pb('./saved/' + saveName, 'ppo/actions-and-internals/categorical/sample/Select')

    print("Model saved with name " + saveName)
