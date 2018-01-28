import os

import gym
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from agent.DQN_agnet import DQNAgent
from utils.utils import Memory, test_agent

if __name__ == "__main__":

    np.random.seed(1024)
    tf.set_random_seed(1024)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))
    # env configure
    MAXSTEP = 500
    convergence_reward = 475

    train_episodes = 1#2000  # 1000          # max number of episodes to learn from
    max_steps = MAXSTEP  # 200                # max steps in an episode
    gamma = 0.99  # future reward discount

    # agent parameters
    state_size = 4
    action_size = 2
    # training process
    train_rewards_list = []
    test_rewards_list = []
    show_every_steps = 100
    # Exploration parameters
    explore_start = 0.9  # exploration probability at start
    explore_stop = 0.01  # minimum S probability
    decay_rate = 0.0001  # expotentional decay rate for exploration prob
    # Network parameters
    hidden_size = 20  # number of units in each Q-network hidden layer
    learning_rate = 0.01  # Q-network learning rate
    # Memory parameters
    memory_size = 10000  # memory capacity
    batch_size = 32  # experience mini-batch size
    pretrain_length = batch_size  # number experiences to pretrain the memory
    memory = Memory(max_size=memory_size)

    # Initialize the simulation
    env = gym.make('CartPole-v1')

    # TODO 指定网络参数和名字
    agent = DQNAgent(env, explore_start, explore_stop, decay_rate, state_size=state_size, action_size=action_size, hidden_size=hidden_size,
                     use_targetQ=False, C=20, use_dueling=False, use_DDQN=False, lr=learning_rate)
    model_name = "nips"

    state = env.reset()
    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        # env.render()
        # Make a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            # Start new episode
            state = env.reset()

        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

    step = 0
    for ep in range(1, train_episodes+1):
        total_reward = 0
        t = 0
        # episode_total
        while t < max_steps:
            step += 1
            # Uncomment this next line to watch the training
            # env.render()
            action = agent.get_action(state)
            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps
                if ep % show_every_steps == 0:
                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(agent.explore_p))
                train_rewards_list.append((ep, total_reward))
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                # Start new episode
                state = env.reset()
            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1
            # Sample mini-batch from memory
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            # Train network

            loss = agent.learn(states, actions, rewards, gamma, next_states)

        test_rewards_list.extend(test_agent(
            agent, env, test_max_steps=convergence_reward+25))
        cur_compute_len = min(100, len(test_rewards_list))
        mean_reward = np.mean(test_rewards_list[len(
            test_rewards_list) - cur_compute_len:])
        print('Episode: {}'.format(ep),
              'Mean test reward: {:.1f}'.format(mean_reward), )
        if mean_reward > convergence_reward:
            print(ep, "solved")
            break
    agent.save(model_name + ".h5")
    np.save(model_name + "_train_rewards.npy", train_rewards_list)
    np.save(model_name + "_test_rewards.npy", test_rewards_list)
