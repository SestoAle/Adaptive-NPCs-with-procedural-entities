from tensorforce.agents import Agent

def create_agents(net, baseline, args, embedding_type='dense_embedding'):

    states = dict()

    if embedding_type == 'dense_embedding':
        states = {
        'global_in': {'shape': (10, 10, 52), 'type': 'float'},
        'local_in': {'shape': (5, 5, 52), 'type': 'float'},
        'local_in_two': {'shape': (3, 3, 52), 'type': 'float'},


        'agent_stats': {'shape': (13), 'num_values': 108, 'type': 'int'},
        'target_stats': {'shape': (4), 'num_values': 32, 'type': 'int'},

        'prev_action': {'shape': (17), 'type': 'float'}
    }
    elif embedding_type == 'transformer':
        states = {
            'global_in': {'shape': (10, 10), 'num_values': 4, 'type': 'int'},
            'local_in': {'shape': (5, 5), 'num_values': 4, 'type': 'int'},
            'local_in_two': {'shape': (3, 3), 'num_values': 4, 'type': 'int'},

            'melee_weapons': {'shape': (20, 44), 'type': 'float'},
            'range_weapons': {'shape': (20, 44), 'type': 'float'},
            'potions': {'shape': (20, 44), 'type': 'float'},

            'global_indices': {'shape': (60), 'type': 'float'},
            'local_indices': {'shape': (60), 'type': 'float'},
            'local_two_indices': {'shape': (60), 'type': 'float'},

            'agent_stats': {'shape': (13), 'num_values': 108, 'type': 'int'},
            'target_stats': {'shape': (4), 'num_values': 32, 'type': 'int'},

            'prev_action': {'shape': (17), 'type': 'float'}
        }


    # Create a Proximal Policy Optimization agent
    agent = Agent.create(
        # Agent type
        agent='ppo',
        # Inputs structure
        states=states,
        # Actions structure
        actions={
            'type': 'int',
            'num_values': 17
        },
        network=net,
        # MemoryModel

        # 10 episodes per update
        batch_size=int(args.update_number),
        # Every 10 episodes
        update_frequency=int(args.update_number) - 5,
        max_episode_timesteps=args.num_timesteps,

        # DistributionModel

        discount=0.99,
        entropy_regularization=0.01,
        likelihood_ratio_clipping=0.2,

        critic_network=baseline,

        critic_optimizer=dict(
            type='multi_step',
            optimizer=dict(
                type='subsampling_step',
                fraction=0.33,
                optimizer=dict(
                    type='adam',
                    learning_rate=5e-4
                )
            ),
            num_steps=10
        ),

        # PPOAgent
        learning_rate=5e-6,

        subsampling_fraction=0.33,
        optimization_steps=20,
        # TODO: check this part
        execution=None,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, saver=None,
        summarizer=None, recorder=None
    )

    return agent, states