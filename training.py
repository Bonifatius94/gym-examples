import random
from typing import Tuple, Protocol
from timeit import timeit

import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer


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
            self.state = self.env.reset()

        return state_before, action, reward, state_after, is_done


class TrainableAgent:
    def __init__(self, nn: tf.keras.Model=None):
        self.nn: tf.keras.Model = self._create_neural_network() if not nn else nn
        self.learn_rate = 0.1
        self.exploration_rate = self.ExponentialDecay(1.0, 0.01, 10000)
        self.gamma = 0.99

    def select_action(self, state: State):
        return reduce(func_compose, [
            lambda: random.random() < self.exploration_rate(),
            lambda explore: random.randint(0, 3) \
                if explore else self._pick_best_action(state)
        ])()

    def update_model(self, sars_exp: SARSExperience):
        # TODO: implement the training step
        pass

    def _pick_best_action(self, state: State) -> Action:
        return np.argmax(self._predict_single(state))

    def _predict_single(self, state: State) -> np.ndarray:
        return self._predict_batch(np.expand_dims(state, axis=0))[0]

    def _predict_batch(self, states: np.ndarray) -> np.ndarray:
        return self.nn(states).numpy()

    @staticmethod
    def _create_neural_network() -> tf.keras.Model:
        model = Sequential([
            InputLayer((1)),
            Dense(128),
            Activation('relu'),
            Dense(256),
            Activation('relu'),
            Dense(4),
        ])

        return model

    class ExponentialDecay:
        def __init__(self, initial_value: float, final_value: float, decay_steps: int):
            self.value = initial_value
            self.final_value = final_value
            self.decay_rate = pow(final_value / initial_value, 1 / decay_steps)
            self.step = 0

        def __call__(self) -> float:
            self.value = self.value * self.decay_rate
            return self.final_value if self.value < self.final_value else self.value


class Agent(Protocol):
    def select_action(self, _: State) -> Action:
        ...

    def update_model(self, _: SARSExperience):
        ...


def func_compose(f1, f2):
    return lambda: f1(f2())


def reduce(red_func, items):
    aggr = items[0]
    for item in items[1:]:
        aggr = red_func(item, aggr)
    return aggr


def train_episode(env: Env, agent: Agent) -> Tuple[float, int]:
    is_done = False
    rewards = 0
    steps = 0

    def update_metrics(sars_exp: SARSExperience) -> SARSExperience:
        nonlocal is_done, rewards, steps
        is_done = sars_exp[4]
        rewards += sars_exp[2]
        steps += 1
        return sars_exp

    while not is_done:
        reduce(func_compose, [
            lambda: agent.select_action(env.state),
            env.apply_action,
            update_metrics,
            agent.update_model
        ])()

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


def run_training():
    env = Env()
    agent = TrainableAgent()

    log_interval = 10
    avg_rewards = AverageMetric()
    avg_steps_per_episode = AverageMetric()

    def log_episode(reward_sum: float, steps: int):
        nonlocal log_interval, avg_rewards, avg_steps_per_episode
        avg_rewards.append(reward_sum)
        avg_steps_per_episode.append(steps)

        if episode % log_interval == 0:
            print(
                f'episode {episode}, '
                f'avg reward {avg_rewards()}, '
                f'avg steps {avg_steps_per_episode()}')
            avg_rewards.reset()
            avg_steps_per_episode.reset()

    print('starting training')

    for episode in range(1, 10001):
        reduce(func_compose, [
            lambda: train_episode(env, agent),
            lambda args: log_episode(args[0], args[1])
        ])()

    print('training finished')


def run_environment_perft():
    class RandomAgent:
        def select_action(self, _: State) -> Action:
            return random.randint(0, 3)

        def update_model(self, _: SARSExperience):
            pass

    def perft(num_episodes: int):
        env = Env()
        agent = RandomAgent()
        for _ in range(num_episodes):
            train_episode(env, agent)

    perft_eps = 1000
    print(f'gym took {timeit(lambda: perft(perft_eps), number=10) / 10} '
          f'seconds for {perft_eps} episodes')


def main():
    run_environment_perft()
    run_training()


if __name__ == '__main__':
    main()
