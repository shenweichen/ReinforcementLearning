from keras.layers import Dense, Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import numpy as np


class DQNAgent:
    """
    nips dqn use_targetQ=False,use_DDQN=False，use_duelling=False
    nature dqn use_targetQ=True,use_DDQN=False，use_duelling=False
    double dqn use_targetQ=True,use_DDQN=True，use_duelling=False
    dueling dqn use_dueling = True
    """

    def __init__(self, env, exp_start, exp_end, decay, state_size, action_size,
                 use_targetQ=False, C=5, use_DDQN=False, use_dueling=False,
                 hidden_size=20, lr=0.0001):
        """

        :param env:
        :param exp_start:
        :param exp_end:
        :param decay:
        :param state_size:
        :param action_size:
        :param use_targetQ:
        :param C:
        :param use_DDQN:
        :param use_dueling:
        :param hidden_size:
        :param lr:
        """
        self.explore_start = exp_start
        self.explore_stop = exp_end
        self.decay_rate = decay
        self.explore_p = exp_start
        self.env = env
        self.use_dueling = use_dueling
        self.update_count = 0
        self.C = C
        self.lr = lr
        self.use_DDQN = use_DDQN
        self.model = self.buildmodel(state_size, action_size, hidden_size)
        self.use_targetQ = use_targetQ
        if use_DDQN:
            if use_targetQ is False:
                raise ValueError("use_targetQ must be True if use_DDQN=True")
        if use_targetQ:
            if self.C <= 0:
                raise ValueError("C must greater than 0 if use_targetQ=True")
            self.targetQ = self.buildmodel(
                state_size, action_size, hidden_size)

    def update_targetQ(self):
        # copy weights from model to target_model
        self.targetQ.set_weights(self.model.get_weights())

    def buildmodel(self, state_size, action_size, hidden_size=64):
        input_state = Input((state_size,))
        # fc1 = Dense(10, activation='tanh')(input_state)
        fc1 = input_state
        if self.use_dueling:
            stream1 = Dense(hidden_size, activation='tanh')(fc1)
            stream2 = Dense(hidden_size, activation='tanh')(fc1)
            value = Dense(1, )(stream1)
            advantage = Dense(action_size)(stream2)
            Q = Lambda(lambda a: a[0] + a[1] - K.mean(a[1],
                                                      axis=1, keepdims=True), )([value, advantage])
        else:
            fc2 = Dense(hidden_size, activation='tanh')(fc1)
            Q = Dense(action_size, )(fc2)
        model = Model(inputs=[input_state], outputs=[Q])
        model.compile(loss="mse", optimizer="adam")  # Adam(self.lr)
        return model

    def clip_mse(self, y_true, y_pred):
        error = K.clip(y_true - y_pred, -1, 1)
        return K.mean(K.square(error), axis=-1)

    def get_action(self, state, explore=True):
        """
        get action from Q-Value
        :param state:
        :param explore:bool
        :return:
        """
        if explore:
            self.explore_p = self.explore_stop + \
                (self.explore_start - self.explore_stop) * \
                np.exp(-self.decay_rate * self.update_count)
            if self.explore_p > np.random.rand():
                # Make a random action
                action = self.env.action_space.sample()
                return action
        state = np.reshape(state, (-1, *state.shape))
        Qs = self.model.predict(state)
        action = np.argmax(Qs[0])
        return action

    def learn(self, states, actions, rewards, gamma, next_states):
        if self.use_targetQ:
            target_Qs = self.targetQ.predict_on_batch(
                next_states)  # use targetQ Net to generate Q-value
        else:
            target_Qs = self.model.predict_on_batch(next_states)

        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0)
        if self.use_DDQN:
            greedy_actions = np.argmax(
                self.model.predict(next_states), axis=1)  # use online model to select greedy actions
            # use targetQ to evaluate action
            targets = rewards + gamma * \
                target_Qs[range(len(greedy_actions)), greedy_actions]
        else:
            targets = rewards + gamma * np.max(target_Qs, axis=1)

        estimate_Qs = self.model.predict(states)
        # update the selected actions value
        estimate_Qs[range(len(actions)), actions] = targets
        cost = self.model.train_on_batch(states, estimate_Qs)
        self.update_count += 1
        if self.use_targetQ and self.update_count % self.C == 0:
            self.update_targetQ()
        return cost

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def load_model(self, path):
        self.model = load_model(path)

    def save(self, path):
        self.model.save(path)
