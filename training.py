import random
from typing import Tuple

import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


State = int
Action = int
SARSExperience = Tuple[State, Action, float, State, bool]


class Env:
    def __init__(self):
        self.env = gym.make('FrozenLake-v1', map_name='4x4', new_step_api=True)
        self.state = self.env.reset()

    def apply_action(self, action: Action) -> SARSExperience:
        state_before = self.state
        result = self.env.step(action)
        state_after, reward, is_done, _, _ = result

        self.state = state_after
        if is_done:
            self.env.reset()

        return state_before, action, reward, state_after, is_done


class ExponentialDecay:
    def __init__(self, initial_value: float, final_value: float, decay_steps: int):
        self.value = initial_value
        self.final_value = final_value
        self.decay_rate = pow(final_value / initial_value, 1 / decay_steps)
        self.step = 0

    def __call__(self) -> float:
        self.value = self.value * self.decay_rate
        return self.final_value if self.value < self.final_value else self.value


class Agent:
    def __init__(self, nn: tf.keras.Model=None):
        self.nn: tf.keras.Model = self._create_neural_network() if not nn else nn
        self.loss_fn = MeanSquaredError()
        self.optimizer = Adam(learning_rate=0.02)
        self.exploration_rate = ExponentialDecay(1.0, 0.03, 50000)
        self.gamma = 0.99

    def select_action(self, state: State):
        explore = random.random() < self.exploration_rate()
        return random.randint(0, 3) if explore else self._pick_best_action(state)

    def update_model(self, sars_exp: SARSExperience):
        state_before, action, reward, state_after, is_done = sars_exp
        q_after = self._predict_single(state_after)
        q_target = np.zeros(4)
        terminal = 1.0 if is_done else self.gamma * max(q_after)
        q_target[action] = reward + terminal

        x = np.expand_dims(np.expand_dims(state_before, axis=0), axis=0)
        y = np.expand_dims(q_target, axis=0)

        with tf.GradientTape() as tape:
            pred = self.nn(x)
            loss_mask = tf.expand_dims(tf.one_hot(np.array(action), 4), axis=0)
            masked_pred = tf.multiply(pred, loss_mask)
            loss = self.loss_fn(y, masked_pred)

        grads = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))

    def _pick_best_action(self, state: State) -> Action:
        return np.argmax(self._predict_single(state))

    def _predict_single(self, state: State) -> np.ndarray:
        return self._predict_batch(np.expand_dims(state, axis=0))[0]

    def _predict_batch(self, states: np.ndarray) -> np.ndarray:
        return self.nn(states).numpy()

    @staticmethod
    def _create_neural_network() -> tf.keras.Model:
        model = Sequential([
            Dense(64),
            Activation('sigmoid'),
            Dense(16),
            Activation('sigmoid'),
            Dense(4),
            Activation('sigmoid'),
        ])

        model.build((None, 1))
        return model


def train_episode(env: Env, agent: Agent) -> Tuple[float, int]:
    is_done = False
    rewards = 0
    steps = 0

    while not is_done:
        action = agent.select_action(env.state)
        sars_exp = env.apply_action(action)
        agent.update_model(sars_exp)
        is_done = sars_exp[4]
        rewards += sars_exp[2]
        steps += 1

    return rewards, steps


class AverageMetric:
    def __init__(self):
        self.data_points = []

    def __call__(self) -> float:
        return sum(self.data_points) / len(self.data_points)

    def append(self, point: float):
        self.data_points.append(point)

    def reset(self):
        self.data_points = []


class RandomAgent:
    def select_action(self, _: State):
        return random.randint(0, 3)

    def update_model(self, sars_exp: SARSExperience):
        pass


def main():
    env = Env()
    agent = Agent()
    log_interval = 100

    print('starting training')
    avg_rewards = AverageMetric()
    avg_steps_per_episode = AverageMetric()

    for episode in range(1, 50001):
        reward_sum, steps = train_episode(env, agent)
        avg_rewards.append(reward_sum)
        avg_steps_per_episode.append(steps)

        if episode % log_interval == 0:
            print(
                f'episode {episode}, '
                f'avg reward {avg_rewards()}, '
                f'avg steps {avg_steps_per_episode()}')
            avg_rewards.reset()
            avg_steps_per_episode.reset()

    print('training finished')


if __name__ == '__main__':
    main()
