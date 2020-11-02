from tensorforce.execution.runner import Runner
import time
from six.moves import xrange
import warnings
from inspect import getargspec
from export_graph import export_pb

import numpy as np

class DeepCrawlRunner(Runner):

    def __init__(self, agent, environment, repeat_actions=1, history=None, id_=0, curriculum = None, reward_model = None,
                 num_policy_updates = 5, num_discriminator_exp = 5, save_experience = False, model_name = None, dems_name = 'dems.pkl'):
        self.mean_entropies = []
        self.std_entropies = []
        # TODO: change this
        self.real_episode_rewards = []
        self.reward_model_loss = []
        self.reward_model_val_loss = []
        self.history = history
        self.curriculum = curriculum
        self.reward_model = reward_model
        self.num_policy_updates = num_policy_updates
        self.num_discriminator_exp = num_discriminator_exp
        self.save_experience = save_experience
        self.model_name = model_name
        self.current_curriculum_step = 0
        self.dems_name = dems_name
        super(DeepCrawlRunner, self).__init__(agent, environment, repeat_actions, history)

    def set_curriculum(self, curriculum, total_timesteps, self_curriculum_learning = False, mode = 'steps'):

        if curriculum == None:
            return None

        if mode == 'steps':
            lessons = np.cumsum(curriculum['thresholds'])

            curriculum_step = 0

            for (index, l) in enumerate(lessons):
                if total_timesteps > l:
                    curriculum_step = index + 1

            self.current_curriculum_step = curriculum_step

        # TODO: DA FARE ASSOLUTAMENTE CURRICULUM CON MEDIA
        elif mode == 'mean':
            if len(self.episode_rewards) <= 100*6:
                self.current_curriculum_step = 0
                pass

            means = []
            for i in range(6):
                mean = np.mean(self.episode_rewards[:-100*(i+1)])
                means.append(mean)

            mean = np.mean(np.asarray(means))
            if mean - curriculum['lessons'][self.current_curriculum_step] < 0.05:
                self.current_curriculum_step += 1

            config = {}
            parameters = curriculum['parameters']
            for (par, value) in parameters.items():
                config[par] = value[self.current_curriculum_step]


        parameters = curriculum['parameters']
        config = {}

        for (par, value) in parameters.items():
            config[par] = value[self.current_curriculum_step]

        # Self curriculum setting:
        # Save the model
        if self_curriculum_learning:
            if curriculum_step > self.current_curriculum_step:
                self.agent.save_model('saved/' + self.model_name, append_timestep=False)
                # Freeze the TensorFlow graph and save .bytes file. All the output layers to fetch must be specified
                if self.environment.with_previous:
                    export_pb('saved/' + self.model_name,
                              'ppo/actions-and-internals/categorical/sample/Select,ppo/actions-and-internals/layered-network/apply/internal_lstm0/apply/stack',export_in_unity=True)
                else:
                    export_pb('saved/' + self.model_name, 'ppo/actions-and-internals/categorical/sample/Select', export_in_unity=True)

        return config

    def run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None, deterministic=False,
            episode_finished=None, summary_report=None, summary_interval=None, timesteps=None, episodes=None, testing=False, sleep=None, fixed_reward_model=False
            ):

        # deprecation warnings
        if timesteps is not None:
            num_timesteps = timesteps
            warnings.warn("WARNING: `timesteps` parameter is deprecated, use `num_timesteps` instead.",
                          category=DeprecationWarning)
        if episodes is not None:
            num_episodes = episodes
            warnings.warn("WARNING: `episodes` parameter is deprecated, use `num_episodes` instead.",
                          category=DeprecationWarning)

        # figure out whether we are using the deprecated way of "episode_finished" reporting
        old_episode_finished = False
        if episode_finished is not None and len(getargspec(episode_finished).args) == 1:
            old_episode_finished = True

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()

        self.agent.reset()

        if num_episodes is not None:
            num_episodes += self.agent.episode

        if num_timesteps is not None:
            num_timesteps += self.agent.timestep

        i = 0

        # Initialize Reward Model
        if self.reward_model is not None and not testing and not fixed_reward_model:
            config = self.set_curriculum(self.curriculum, np.sum(self.episode_timesteps))
            self.environment.set_config(config)
            y = input('Do you want to create new demonstrations? [y/n] ')
            if y == 'y':
                dems, vals = self.reward_model.create_demonstrations(env=self.environment)
            else:
                print('Loading demonstrations...')
                dems, vals = self.reward_model.load_demonstrations(self.dems_name)

            print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")
            print('and ' + str(len(vals['obs'])) + " timesteps in these validations.")
        
        policy_buffer = {
            'obs': [],
            'obs_n': [],
            'acts': [],
            'probs': []
        }

        policy_traj = {
            'obs': [],
            'obs_n': [],
            'acts': [],
            'probs': []
        }

        policy_episodes = []

        if self.reward_model is not None and not testing and not fixed_reward_model:
            self.get_experience_(policy_traj, max_episode_timesteps, self.num_policy_updates / self.agent.update_mode['frequency'] * self.num_discriminator_exp)

        # episode loop
        while True:
            episode_start_time = time.time()
            # Set the correct curriculum phase
            config = self.set_curriculum(self.curriculum, np.sum(self.episode_timesteps))

            if i ==0:
               print(config)
            i=1
            self.environment.set_config(config)

            state = self.environment.reset()
            self.agent.reset()

            # Initialize utility buffers
            self.local_entropies = []

            # Update global counters.
            self.global_episode = self.agent.episode  # global value (across all agents)
            self.global_timestep = self.agent.timestep  # global value (across all agents)

            episode_reward = 0
            real_episode_reward = 0
            self.current_timestep = 0

            # IRL setting
            # TODO: move this to reward_model ?
            # If reward_model, update the buffer to train it
            if self.reward_model is not None and not testing and not fixed_reward_model:
                if self.global_episode % self.num_policy_updates == 0:

                    if self.global_episode != 0:
                        for i in range(self.num_discriminator_exp):
                            policy_traj = self.get_experience(max_episode_timesteps, 1)
                            policy_episodes.append(policy_traj)
                        policy_traj = self.policy_episode_to_policy_traj(policy_episodes)

                    self.add_to_poilicy_buffer(policy_buffer, policy_traj)
                    # loss, val_loss = self.reward_model.train_step(self.reward_model.expert_traj, policy_traj)
                    loss, val_loss = self.reward_model.fit(self.reward_model.expert_traj, policy_buffer)
                    self.reward_model_loss.append(loss)
                    self.reward_model_val_loss.append(val_loss)
                    policy_traj = {
                        'obs': [],
                        'obs_n': [],
                        'acts': [],
                        'probs': []
                    }
                    policy_episodes = []

            irl_states = []
            irl_states.append(state)
            irl_acts = []
            irl_probs = []

            # time step (within episode) loop
            while True:
                action, fetches = self.agent.act(states=state, deterministic=deterministic, fetch_tensors=['probabilities'])
                probs = fetches['probabilities']
                self.environment.add_probs(probs)
                self.local_entropies.append(self.environment.get_last_entropy())

                reward = 0
                for _ in xrange(self.repeat_actions):
                    state_n, terminal, step_reward = self.environment.execute(action=action)
                    real_episode_reward += step_reward

                    # Compute the reward from reward model
                    if self.reward_model is not None:
                        #reward_from_model = self.reward_model.forward([state], [action])
                        # reward_from_model = self.reward_model.forward([state], [action])
                        probs = np.squeeze(probs)
                        if not fixed_reward_model:
                            reward_from_model = self.reward_model.eval_discriminator([state], [state_n], [probs[action]], [action])
                        else:
                            reward_from_model = self.reward_model.eval([state], [action])
                        # Normalize reward
                        #reward_from_model = [reward_from_model]
                        #reward_from_model = reward_from_model.cpu().detach().numpy()
                        if not fixed_reward_model:
                            self.reward_model.push_reward(reward_from_model)
                        else:
                            self.reward_model.push_reward(reward_from_model)

                        reward_from_model = self.reward_model.normalize_rewards(reward_from_model)
                        reward_from_model = np.squeeze(reward_from_model)
                        
                        #step_reward = self.environment.filter_reward(step_reward)
                        #step_reward += reward_from_model
                        step_reward = reward_from_model

                        irl_states.append(state_n)
                        irl_acts.append(action)
                        irl_probs.append(probs[action])

                    state = state_n

                    reward += step_reward
                    if terminal:
                        break

                if max_episode_timesteps is not None and self.current_timestep >= max_episode_timesteps:
                    terminal = True

                if not testing:
                    self.agent.observe(terminal=terminal, reward=reward)

                self.global_timestep += 1
                self.current_timestep += 1
                episode_reward += reward

                # Update utility buffers
                # states.append(state)
                # acts.append(action)
                # all_probs.append(np.squeeze(probs)[action])

                if terminal or self.agent.should_stop():  # TODO: should_stop also terminate?
                    break

                if sleep is not None:
                    time.sleep(sleep)

            # Update our episode stats.
            time_passed = time.time() - episode_start_time
            self.episode_rewards.append(episode_reward)
            self.real_episode_rewards.append(real_episode_reward)
            self.episode_timesteps.append(self.current_timestep)
            self.episode_times.append(time_passed)
            self.mean_entropies.append(np.mean(self.local_entropies))
            self.std_entropies.append(np.std(self.local_entropies))

            if self.reward_model is not None and not testing and not fixed_reward_model: #and\
                    #self.global_episode % self.agent.update_mode['frequency'] == 0:
                #self.get_experience_(policy_traj, max_episode_timesteps, self.num_discriminator_exp)
                policy_traj['obs'] = irl_states[:-1]
                policy_traj['obs_n'] = irl_states[1:]
                policy_traj['acts'] = irl_acts
                policy_traj['probs'] = irl_probs
                policy_episodes.append(policy_traj)
                policy_traj = {
                    'obs': [],
                    'obs_n': [],
                    'acts': [],
                    'probs': []
                }

            self.global_episode += 1

            # Check, whether we should stop this run.
            if episode_finished is not None:
                # deprecated way (passing in only runner object):
                if old_episode_finished:
                    if not episode_finished(self):
                        break
                # new unified way (passing in BaseRunner AND some worker ID):
                elif not episode_finished(self, self.id):
                    break
            if (num_episodes is not None and self.global_episode >= num_episodes) or \
                    (num_timesteps is not None and self.global_timestep >= num_timesteps) or \
                    self.agent.should_stop():
                break

    def reset(self, history=None):
        super(DeepCrawlRunner, self).reset(history)
        if(history != None):
            self.std_entropies = history.get("std_entropies", list())
            self.mean_entropies = history.get("mean_entropies", list())
            self.real_episode_rewards = history.get("real_episode_rewards", list())
            self.reward_model_loss = history.get("reward_model_loss", list())


    # IRL settings
    def get_experience(self, max_episode_timesteps, num_discriminator_exp = None, verbose = False):

        if num_discriminator_exp == None:
            num_discriminator_exp = self.num_policy_updates

        policy_traj = {
            'obs': [],
            'obs_n': [],
            'acts': [],
            'probs': []
        }

        # For policy update number
        for ep in range(num_discriminator_exp):
            states = []
            probs = []
            actions = []
            state = self.environment.reset()
            states.append(state)

            step = 0
            # While the episode si not finished
            reward = 0
            while True:
                step += 1
                # Get the experiences that are not saved in the agent
                action, fetch = self.agent.act(states=state, deterministic=False, independent=True,
                                               fetch_tensors=['probabilities'])
                c_probs = np.squeeze(fetch['probabilities'])
                state, terminal, step_reward = self.environment.execute(action=action)

                reward += step_reward

                states.append(state)
                actions.append(action)
                probs.append(c_probs[action])

                if terminal or step >= max_episode_timesteps:
                    break

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

            # Saved the last episode experiences
            policy_traj['obs'].extend(states[:-1])
            policy_traj['obs_n'].extend(states[1:])
            policy_traj['acts'].extend(actions)
            policy_traj['probs'].extend(probs)

        # Return all the experience
        return policy_traj

    # IRL settings
    def get_experience_(self, policy_traj, max_episode_timesteps, num_discriminator_exp=None, verbose=False, deterministic=False,
                           testing=False):
            if num_discriminator_exp == None:
                num_discriminator_exp = self.num_policy_updates

            # For policy update number
            ep = 0
            while ep < num_discriminator_exp:
                states = []
                probs = []
                actions = []
                state = self.environment.reset()
                states.append(state)

                step = 0
                # While the episode si not finished
                reward = 0
                while True:
                    step += 1
                    # Get the experiences that are not saved in the agent
                    action, fetch = self.agent.act(states=state, deterministic=deterministic, independent=True,
                                                   fetch_tensors=['probabilities'])
                    c_probs = np.squeeze(fetch['probabilities'])
                    state, terminal, step_reward = self.environment.execute(action=action)

                    reward += step_reward

                    states.append(state)
                    actions.append(action)
                    probs.append(c_probs[action])

                    if terminal or step >= max_episode_timesteps:
                        break

                if verbose:
                    print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))
                # Saved the last episode experiences
                ep += 1
                policy_traj['obs'].extend(states[:-1])
                policy_traj['obs_n'].extend(states[1:])
                policy_traj['acts'].extend(actions)
                policy_traj['probs'].extend(probs)

            # Return all the experience
            return policy_traj

    def add_to_poilicy_buffer(self, policy_buffer, new_policy_buffer, buffer_length = 10000*20*30):

        if len(policy_buffer['obs']) + len(new_policy_buffer['obs']) > buffer_length:
            diff = len(policy_buffer['obs']) + len(new_policy_buffer['obs']) - buffer_length
            policy_buffer['obs'] = policy_buffer['obs'][diff:]
            policy_buffer['obs_n'] = policy_buffer['obs_n'][diff:]
            policy_buffer['acts'] = policy_buffer['acts'][diff:]
            policy_buffer['probs'] = policy_buffer['probs'][diff:]

        policy_buffer['obs'].extend(new_policy_buffer['obs'])
        policy_buffer['obs_n'].extend(new_policy_buffer['obs_n'])
        policy_buffer['acts'].extend(new_policy_buffer['acts'])
        policy_buffer['probs'].extend(new_policy_buffer['probs'])


    def policy_episode_to_policy_traj(self, policy_episodes):
        policy_episodes = policy_episodes[self.num_discriminator_exp:]
        policy_traj = {
            'obs': [],
            'obs_n': [],
            'acts': [],
            'probs': []
        }
        for episode in policy_episodes:
            policy_traj['obs'].extend(episode['obs'])
            policy_traj['obs_n'].extend(episode['obs_n'])
            policy_traj['acts'].extend(episode['acts'])
            policy_traj['probs'].extend(episode['probs'])

        return policy_traj
