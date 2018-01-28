# CartPole-v1

## 实验结果

|model|episode to solve|test result(100 episode Mean Std Max Min)|
|--|--|--|
|nips dqn|476|**499.0 0.0 499 499**|
|nips dqn with dueling|353|**499.0 0.0 499 499**|
|nature dqn|320|492.6 29 499 315|
|nature dqn with dueling|**242**|457.7 18 499 411|
|double dqn|309|**499.0 0.0 499 499**|
|double dqn with dueling|302|495.0 11 499 446|
## 初步结论

在`CartPole`环境下，几种方法的差异并不是很大。  
事实上，实验还受到了随机初始化的影响，在某些情况下一开始的时候就可能获得很好的效果。
- solve speed  
`double > nature > nips`,同时`dueling`结构有助于提升速度
- model performence  
 `double = nips with dueling = nips >double with dueling > nature > nature with dueling`  
 从性能表现和稳定性来看，`double dqn,nipsDQN`以及引入`dueling`结构的`nipsDQN`表现要好过其余模型

**其实仅仅通过这一个场景很难比较出不同方法之间的优劣，每种方法都有各自使用的场景，还需要继续深入学习和理解。**
## 相关配置参数

```Python
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
hidden_size = 20  # number of units in each Q-network hidden layer Now only use one hidden layer
learning_rate = 0.001  # Q-network learning rate
# Memory parameters
memory_size = 10000  # memory capacity
batch_size = 32  # experience mini-batch size
pretrain_length = batch_size  # number experiences to pretrain the memory
```

## 相关曲线图表

![loading](https://github.com/shenweichen/DeepRL/raw/master/experiment/CartPole-v1/NIPS_DQN.png)
![loading](https://github.com/shenweichen/DeepRL/raw/master/experiment/CartPole-v1/NIPS_DQN_with_Dueling.png)
![loading](https://github.com/shenweichen/DeepRL/raw/master/experiment/CartPole-v1/Nature_DQN.png)
![loading](https://github.com/shenweichen/DeepRL/raw/master/experiment/CartPole-v1/Nature_DQN_with_Dueling.png)
![loading](https://github.com/shenweichen/DeepRL/raw/master/experiment/CartPole-v1/Double_DQN.png)
![loading](https://github.com/shenweichen/DeepRL/raw/master/experiment/CartPole-v1/Double_DQN_with_Dueling.png)