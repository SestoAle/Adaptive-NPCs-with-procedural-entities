from tensorforce.execution.runner import Runner
import numpy as np
import time

class DeepCrawlRunner(Runner):

    def __init__(self, agent, environment, max_episode_timesteps=None, history=None, curriculum = None):
        self.mean_entropies = []
        self.std_entropies = []
        self.history = history
        self.curriculum = curriculum
        self.i = 0
        self.unity_env = environment
        # DeepCrawl
        if not isinstance(environment,list):
            environment = self.unity_env
            environments = None
            self.unity_env = [self.unity_env]
        else:
            environment = None
            environments = self.unity_env

        self.local_entropies = np.empty((len(self.unity_env), 0)).tolist()

        super(DeepCrawlRunner, self).__init__(agent, environment=environment, environments=environments, max_episode_timesteps=100)

        # DeepCrawl
        for env in self.unity_env:
            config = self.set_curriculum(self.curriculum, np.sum(self.history['episode_timesteps']))
            if self.i == 0:
                print(config)
            self.i = 1
            env.set_config(config)
            # DeepCrawl

        #self.reset(history)

    # DeepCrawl: Update curriculum
    def set_curriculum(self, curriculum, total_timesteps, self_curriculum_learning=False, mode='steps'):

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
            if len(self.episode_rewards) <= 100 * 6:
                self.current_curriculum_step = 0
                pass

            means = []
            for i in range(6):
                mean = np.mean(self.episode_rewards[:-100 * (i + 1)])
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
        # TODO: FARE SELF CURRICULUM PLAY
        if self_curriculum_learning:
            if curriculum_step > self.current_curriculum_step:
                self.agent.save_model('saved/' + self.model_name, append_timestep=False)
                # Freeze the TensorFlow graph and save .bytes file. All the output layers to fetch must be specified
                if self.environment.with_previous:
                    export_pb('saved/' + self.model_name,
                              'ppo/actions-and-internals/categorical/sample/Select,ppo/actions-and-internals/layered-network/apply/internal_lstm0/apply/stack',
                              export_in_unity=True)
                else:
                    export_pb('saved/' + self.model_name, 'ppo/actions-and-internals/categorical/sample/Select',
                              export_in_unity=True)

        return config

    def handle_act(self, parallel):
        if self.batch_agent_calls:
            self.environments[parallel].start_execute(actions=self.actions[parallel])

        else:
            agent_start = time.time()

            # DeepCrawl
            query = ['action-distribution-probabilities']
            actions, probs = self.agent.act(states=self.states[parallel], parallel=parallel, query=query)
            probs = probs[0]
            self.unity_env[parallel].add_probs(probs)
            self.local_entropies[parallel].append(self.unity_env[parallel].get_last_entropy())
            # DeepCrawl
            self.episode_agent_second[parallel] += time.time() - agent_start

            self.environments[parallel].start_execute(actions=actions)

        # Update episode statistics
        self.episode_timestep[parallel] += 1

        # Maximum number of timesteps or timestep callback (after counter increment!)
        self.timesteps += 1
        if (
            (self.episode_timestep[parallel] % self.callback_timestep_frequency == 0 and not self.callback(self)) or
            self.timesteps >= self.num_timesteps
        ):
            self.terminate = 2

    def handle_terminal(self, parallel):
        # Update experiment statistics
        self.episode_rewards.append(self.episode_reward[parallel])
        self.episode_timesteps.append(self.episode_timestep[parallel])
        self.episode_seconds.append(time.time() - self.episode_start[parallel])
        self.episode_agent_seconds.append(self.episode_agent_second[parallel])
        # DeepCrawl
        self.mean_entropies.append(np.mean(self.local_entropies[parallel]))
        self.std_entropies.append(np.std(self.local_entropies[parallel]))
        self.update_history()
        # DeepCrawl

        # Maximum number of episodes or episode callback (after counter increment!)
        self.episodes += 1
        if self.terminate == 0 and ((
            self.episodes % self.callback_episode_frequency == 0 and
            not self.callback(self, parallel)
        ) or self.episodes >= self.num_episodes):
            self.terminate = 1

        # Reset episode statistics
        self.episode_reward[parallel] = 0.0
        self.episode_timestep[parallel] = 0
        self.episode_agent_second[parallel] = 0.0
        self.episode_start[parallel] = time.time()

        # Reset environment
        if self.terminate == 0 and not self.sync_episodes:
            self.terminals[parallel] = -1
            # DeepCrawl
            # Set curriculum configuration
            for env in self.unity_env:
                config = self.set_curriculum(self.curriculum, np.sum(self.history['episode_timesteps']))
                if self.i == 0:
                    print(config)
                self.i = 1
                env.set_config(config)
            # DeepCrawl
            self.environments[parallel].start_reset()

    def update_history(self):
        self.history["episode_rewards"].extend(self.episode_rewards)
        self.history["episode_timesteps"].extend(self.episode_timesteps)
        self.history["mean_entropies"].extend(self.mean_entropies)
        self.history["std_entropies"].extend(self.std_entropies)
        self.reset()

    def reset(self):
        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.std_entropies = list()
        self.mean_entropies = list()
        #self.real_episode_rewards = history.get("real_episode_rewards", list())
        self.reward_model_loss = list()
