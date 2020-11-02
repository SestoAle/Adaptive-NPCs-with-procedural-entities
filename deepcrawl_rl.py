import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np

import tensorflow as tf
import time
import json


from unity_env_wrapper import UnityEnvWrapper
from deepcrawl_runner import DeepCrawlRunner
from agents.agents import create_agents

import argparse

import datetime



use_cuda = True

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''---------'''
'''Arguments'''
'''---------'''

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model-name', help="The name of the model", default="")
parser.add_argument('-et', '--embedding-type', help="The type of the embedding module you want to use", default="transformer")
parser.add_argument('-ne', '--num-episodes', help="Specify the number of episodes after which the environment is restarted", default=None)
parser.add_argument('-wk', '--worker-id', help="The id for the worker", default=0)

parser.add_argument('-mt', '--num-timesteps', help="Max timesteps of the agent", default=100)
parser.add_argument('-ls', '--lstm', help="If the net has an lstm", default=True)
parser.add_argument('-un', '--update-number', help="Update dict", default=10)
parser.add_argument('-ws', '--with-stats', help="Update dict", default=True)
parser.add_argument('-sa', '--save', help="Number of episodes for saving models", default=3000)

args = parser.parse_args()


'''--------------------'''
'''Algorithm Parameters'''
'''--------------------'''

# Import net structures and game name based on the embedding type
if args.embedding_type is 'dense_embedding':
    from net_structures.net_structures import dense_embedding_net as net
    from net_structures.net_structures import dense_embedding_baseline as baseline

    args.game_name = 'envs/DeepCrawl-Dense-Embedding'
elif args.embedding_type is 'transformer':
    from net_structures.net_structures import transformer_net as net
    from net_structures.net_structures import transformer_baseline as baseline

    args.game_name = 'envs/DeepCrawl-Transformer'

# Create agent and state spec based on embedding mode
agent, states_spec = create_agents(net, baseline, args, embedding_type=args.embedding_type)

# Work ID of the environment. To use more environments in parallel, use
# different ids
work_id = int(args.worker_id)

# Number of episodes of a single run
num_episodes = args.num_episodes
# Number of timesteps within an episode
num_timesteps = args.num_timesteps
lstm = args.lstm

# Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
curriculum = {
    'current_step': 0,
    'thresholds': [1e6, 0.8e6, 1e6, 1e6],
    'parameters':
        {
            'minTargetHp': [1,10,10,10,15],
            'maxTargetHp': [1,10,20,20,20],
            'minAgentHp': [15,10,5,5,5],
            'maxAgentHp': [20,20,20,20,20],
            'minNumLoot': [0.2,0.2,0.2,0.08,0.04],
            'maxNumLoot': [0.2,0.2,0.2,0.2,0.2],
            'numActions': [17,17,17,17,17],
            # Agent statistics
            'agentAtk': [3,3,3,3,3],
            'agentDef': [3,3,3,3,3],
	        'agentDes': [3,3,3,3,3],
        }
}

model_name = args.model_name

# Name of the enviroment to load. To use Unity Editor, must be None
game_name = args.game_name

'''-----------------'''
'''Run the algorithm'''
'''-----------------'''

use_model = None

if model_name == "" or model_name == " " or model_name == None:

    while(use_model != 'y' and use_model != 'n'):
        use_model = input('Do you want to use a previous saved model? [y/n] ')

    if (use_model == 'y'):
        model_name = input('Name of the model: ')
        try:
            with open("arrays/" + model_name + ".json") as f:
                history = json.load(f)
        except:
            history = {
                "episode_rewards": list(),
                "episode_timesteps": list(),
                "mean_entropies": list(),
                "std_entropies": list(),
            }
    else:
        model_name = input('Specify the name to save the model: ')
        history = {
            "episode_rewards": list(),
            "episode_timesteps": list(),
            "mean_entropies": list(),
            "std_entropies": list(),
        }

start_time = time.time()

environment = None

# Callback function printing episode statistics
def episode_finished(r, c, episode = 100):
    # Print agent statistics every 100 episodes
    if(len(r.history['episode_rewards']) % episode == 0):
        print('Average cumulative reward for ' + str(episode) + ' episodes @ episode ' + str(len(r.history['episode_rewards'])) + ': ' + str(np.mean(r.history['episode_rewards'][-episode:])))
        print('The agent made ' + str(sum(r.history['episode_timesteps'])) + ' steps so far')
        timer(start_time, time.time())
        reward = 0.0

    # If num_episodes is not defined, save the model every 3000 episodes
    if(num_episodes == None):
        save = int(args.save)
    else:
        save = num_episodes

    if(len(r.history['episode_rewards']) % save == 0):
      save_model(r)

    return True

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def save_model(runner):
    # Save the runner statistics
    # Save the model and the runner statistics
    runner.agent.save(directory='saved', filename=model_name)

    history = runner.history

    json_str = json.dumps(history)
    f = open("arrays/" + model_name + ".json", "w")
    f.write(json_str)
    f.close()

    runner.reset()

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
            agent.restore(directory, model_name)

        # Open the environment with all the desired flags
        environment = UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time()),
                                          worker_id=work_id, with_stats=args.with_stats, size_stats=17,
                                          size_global=10, agent_separate=False, with_class=False, with_hp=False,
                                          with_previous=lstm, verbose=False, manual_input=False,
                                          embedding_type=args.embedding_type)

        environment.set_states(states_spec)

        # Create the runner to run the algorithm
        runner = DeepCrawlRunner(agent=agent, max_episode_timesteps=args.num_timesteps, environment=environment,
                                 history=history, curriculum=curriculum)

        runner.run(num_episodes=num_episodes, callback=episode_finished, use_tqdm=False)

        use_model = 'y'

finally:

    '''--------------'''
    '''End of the run'''
    '''--------------'''

    # Save the model and the runner statistics
    if model_name == "" or model_name == " " or model_name == None:
        saveName = input('Do you want to specify a save name? [y/n] ')
        if(saveName == 'y'):
            saveName = input('Specify the name ')
        else:
            saveName = 'Model'
    else:
        saveName = model_name

    save_model(runner)

    # Close the runner
    runner.close()

    print("Model saved with name " + saveName)
