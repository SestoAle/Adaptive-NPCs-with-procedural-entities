import matplotlib.pyplot as plt
import json
import numpy as np

import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--models-name', help="The name of the model", default='dc22_schifo')
parser.add_argument('-nm', '--num-mean', help="The number of the episode to compute the mean", default=100)
parser.add_argument('-mr', '--num-mean-reward-loss', help="Same as nm, for reward loss", default=10)
parser.add_argument('-sp', '--save-plot', help="If true save the plot in folder saved_plot", default=False)
parser.add_argument('-ep', '--episodes', help="Number of the episodes to observe", default=5000)
parser.add_argument('-xa', '--x-axis', help="Number of the episodes to observe", default='episodes')
args = parser.parse_args()

models_name = args.models_name
while models_name == "" or models_name == " " or models_name == None:
    model_name = input('Insert model name: ')

models_name = models_name.split(",")

histories = []
filenames = []
for model_name in models_name:
    path = glob.glob("arrays/" + model_name + ".json")
    for filename in path:
        with open(filename, 'r') as f:
            histories.append(json.load(f))
            filenames.append(filename)

episodes = args.episodes
if episodes is not None:
    episodes = int(episodes)

print(filenames)
for (index, filename) in enumerate(filenames):
    models_name[index] = filename.replace('arrays/', '').replace('.json', '')

episodes_rewards = []
means_entropies = []
episodes_successes = []
reward_model_losses = []
timestepss = []

for history in histories:

    episodes_reward = np.asarray(history.get("episode_rewards", list()))
    tot_episodes = len(episodes_reward)
    episodes_reward = episodes_reward[:episodes]
    waste = np.alen(episodes_reward)%args.num_mean
    waste = -np.alen(episodes_reward) if waste == 0 else waste
    episodes_reward = episodes_reward[:-waste]
    mean_entropies = np.asarray(history.get("mean_entropies", list()))[:episodes][:-waste]
    std_entropies = np.asarray(history.get("std_entropies", list()))[:episodes][:-waste]
    episodes_success = episodes_reward > 0
    episodes_timesteps = np.asarray(history.get("episode_timesteps", list()))[:episodes][:-waste]
    timesteps = np.asarray(history.get("episode_timesteps", list()))[:episodes][:-waste]

    reward_model_loss = np.asarray(history.get("reward_model_loss", list()))
    tot_updates = len(reward_model_loss)
    if tot_updates > 0:
        num_ep_for_update = int(tot_episodes/tot_updates)
        loss_episodes = int(len(episodes_reward)/num_ep_for_update)
        reward_model_loss = reward_model_loss[:loss_episodes]
        waste_reward_model_loss = np.alen(reward_model_loss)%args.num_mean_reward_loss
        waste_reward_model_loss = -np.alen(reward_model_loss) if waste_reward_model_loss == 0 else waste_reward_model_loss
        reward_model_loss = reward_model_loss[:-waste_reward_model_loss]
    cum_timesteps = np.cumsum(timesteps)

    episodes_rewards.append(episodes_reward)
    means_entropies.append(mean_entropies)
    episodes_successes.append(episodes_success)
    reward_model_losses.append(reward_model_loss)
    timestepss.append(timesteps)

num_mean = int(args.num_mean)
num_mean_reward_loss = int(args.num_mean_reward_loss)
save_plot = bool(args.save_plot)

print("Mean of " + str(num_mean) + " episodes")

model_name = ''
for name in models_name:
    model_name += (name + '_')


plt.figure(1)
plt.title("Reward")
nums_episodes = []
for episodes_reward, model_name, timesteps in zip(episodes_rewards, models_name, timestepss):
    num_episodes = np.asarray(
        range(1, np.size(np.mean(episodes_reward.reshape(-1, num_mean), axis=1)) + 1)) * num_mean

    nums_episodes.append(num_episodes)

    if args.x_axis == 'timesteps':
        x = np.mean(np.cumsum(timesteps).reshape(-1, num_mean), axis=1)
    else:
        x = num_episodes
    plt.plot(x, np.mean(episodes_reward.reshape(-1, num_mean), axis=1))

plt.legend(models_name)
plt.xlabel("Episodes")
plt.ylabel("Mean Reward")
if save_plot:
    plt.savefig("saved_plots/" + model_name + "_reward.png", dpi=300)

plt.figure(2)
plt.title("Entropy")
for (mean_entropies, num_episodes, timesteps) in zip(means_entropies, nums_episodes, timestepss):
    if args.x_axis == 'timesteps':
        x = np.mean(np.cumsum(timesteps).reshape(-1, num_mean), axis=1)
    else:
        x = num_episodes
    plt.plot(x, np.mean(mean_entropies.reshape(-1, num_mean), axis=1))
plt.legend(models_name)
plt.xlabel("Episodes")
plt.ylabel("Mean Entropy")
if save_plot:
    plt.savefig("saved_plots/" + model_name + "_entropy.png", dpi=300)

plt.figure(3)
plt.title("Success")
for (episodes_success, num_episodes, timesteps) in zip(episodes_successes, nums_episodes, timestepss):
    if args.x_axis == 'timesteps':
        x = np.mean(np.cumsum(timesteps).reshape(-1, num_mean), axis=1)
    else:
        x = num_episodes
    plt.plot(x, np.mean(episodes_success.reshape(-1, num_mean), axis=1))
plt.legend(models_name)
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
if save_plot:
    plt.savefig("saved_plots/" + model_name + "_success.png", dpi=300)

for reward_model_loss in reward_model_losses:
    if len(reward_model_loss) > 0:
        plt.figure(4)
        plt.title("Reward Loss")
        reward_model_loss = np.mean(reward_model_loss.reshape(-1, num_mean_reward_loss), axis=1)
        num_reward_updates = np.asarray(
            range(len(reward_model_loss)))
        num_reward_updates = num_reward_updates*int(len(episodes_reward)/len(reward_model_loss))
        plt.plot(num_reward_updates, reward_model_loss)
        plt.legend(models_name)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        if save_plot:
            plt.savefig("saved_plots/" + model_name + "_reward_model_loss.png", dpi=300)


for timesteps, episodes_reward, model_name in zip(timestepss, episodes_rewards, models_name):
    print(model_name + ' max reward: ' + str(np.max(np.mean(episodes_reward.reshape(-1, num_mean), axis=1))))
    print("Number of timesteps: " + str(np.sum(timesteps)))
    print("Number of episodes: " + str(np.size(episodes_reward)))

plt.show()