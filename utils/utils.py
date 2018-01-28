from collections import deque
import numpy as np


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


def test_agent(agent, env, test_max_steps, test_episodes=1, render=False):
    """

    :param agent:
    :param env:
    :param test_max_steps: 每轮最大step
    :param test_episodes: 测试轮数
    :param render: bool 是否开启显示
    :return:
    """
    reward_list = []
    state = env.reset()
    for ep in range(1, test_episodes+1):
        t = 0
        while t < test_max_steps:
            if render:
                env.render()
            action = agent.get_action(state, explore=False)
            next_state, reward, done, _ = env.step(action)
            if done:
                state = env.reset()
                break
            else:
                state = next_state
                t += 1
        reward_list.append(t)
    return reward_list
