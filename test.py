import os

import gym
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from agent.DQN_agnet import DQNAgent
from utils.utils import Memory, test_agent

if __name__ == '__main__':
    np.random.seed(1024)
    tf.set_random_seed(1024)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))
    # env configure
    MAXSTEP = 500
    convergence_reward = 475

    train_episodes = 2000  # 1000          # max number of episodes to learn from
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

    # TODO 指定网络参数和模型名字
    agent = DQNAgent(env, explore_start, explore_stop, decay_rate, state_size=state_size, action_size=action_size,
                     hidden_size=hidden_size,
                     use_targetQ=True, C=20, use_dueling=False, lr=learning_rate)
    model_name = "nips"
    agent.load_model(model_name+'.h5')
    ans = test_agent(agent, env, 500, 10, False)
    ans = np.array(ans)
    print('Mean: {:.1f}'.format(ans.mean()),
          'Std: {:.1f}'.format(ans.std()),
          'Max: {:.1f}'.format(ans.max()),
          'Min: {:.1f}'.format(ans.min()))
