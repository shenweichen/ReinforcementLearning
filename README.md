# DeepRL
对最近看的一些强化学习方法进行记录
# 使用说明
 1. 运行`solve.py`，当满足问题解决条件后，会自动保存模型。
 2. 运行`test.py`，加载保存的模型进行测试。
# 实验说明
1. [Cart-Pole-v1](http://gym.openai.com/envs/CartPole-v1/)  
这是一个比较简单的测试环境，输入状态为4维的连续值组成的向量，输出动作为2个离散动作。
在该环境中，小车能够连续100局保持大于475个step即为解决。  
[**实验结果和相关统计图表**](./experiment/CartPole-v1/results.md)。
# 方法介绍

## Value-Based

### Q-Learning

传统的Q-Learning方法对于每一个(state,action) 对，需要维护其价值。那么当state的维度升高时，就会导致维度灾难，例如对于一个输入是原始图像数据的环境来说，假设输入是<img src="https://latex.codecogs.com/png.latex?200*200" title="200*200" />的像素图片，每个像素有256种取值，那么总共的状态数为<img src="https://latex.codecogs.com/png.latex?256^{200*200}" title="256^{200*200}" />种。这是一个很大数字，会导致存储Q-Tabel消耗巨大的空间，同时Q-Value也将变得非常稀疏，导致学习难以进行。

### Deep Q-Leaning

引入神经网络来对Q(s,a)进行近似表示。神经网络输入为state_size维度的向量,输出为action_size维度的向量，每一维代表一个Q(s,a)值。根据DQN的输出，我们可以获得在当前state下每一个action的Q-Value,根据Q-Value来选择当前的action。

这里简要介绍几种经典的DQN方法，具体细节请查阅相关论文
- Nips DQN

    > Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    - 提出了end2end的DQN学习方法(针对图像输入，使用CNN提取图像信息)
    - 引入经验回放  
       1. 每个step的经验可能被使用多次，提高数据利用率
       2. 连续若干step的经验存在着强相关性，从经验池中随机选择样本有助于打破相关性减少更新的方差
       
- Nature DQN
    > Human-level control through deep reinforcement learning, Mnih et al., 2015
    - 引入了Target Q网络，在计算Target Q值(相当于DQN更新时的label)的时候使用专门的TargetQ网络计算，减少目标Q值与当前Q值得相关性。每隔若干step使用当前Q网络的参数更新TargetQ网络。
    - clip error
    提出将error截断在-1,1之间，提升稳定性
- Double DQN
    >  Deep Reinforcement Learning with Double Q-learning, van Hasselt et al., 2015
    - 主要提出在计算Target Q值的时候，使用当前网络选择贪婪策略，然后使用Target Q网络来评估策略。这样有利于减少由于直接使用max(Target Q)值带来的过度估计。
    - 注意和Double Q-Learning进行区分(Double Q-Learning是交替交换两个Q的角色进行更新，而Double DQN依然是若干step使用当前网络更新TargetQ网络)
- Dueling DQN
    >  Dueling Network Architectures for Deep Reinforcement Learning, Wang et al., 2016
    - 在某些情况下，我们更关注agent所处的state如何，而不是其在该state下做出的action。
    - 主要提出一种duelling结构，将Q网络分解为Value和Advantage,分别学习当前state的价值和在当前状态下各个action的相对价值。最后将两个通道的输出相加得到估计得Q-Value
    - 实现过程中由于Q = V+A,对V和A同时加减一个常量不影响Q值的大小,为了能得到固定的V和A，对A减去所有A的均值。

## Policy-Based

待补充

## Actor-Critic

待补充