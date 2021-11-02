import numpy as np

import tqdm
import gym


class ToyEnv(gym.Env):

    def __init__(self, problem='4state'):

        self.action_space = gym.spaces.Discrete(2)

        self._state = None
        self._diamond = None
        self._diamond_2 = None

        self.fn = {'4state': self._4_state_problem, '5state': self._5_state_problem}[problem]

        pos_space = {'4state': gym.spaces.Discrete(4), '5state': gym.spaces.Discrete(5)}[problem]

        self.observation_space = gym.spaces.Dict({"position": pos_space,
                                                  "diamond": gym.spaces.Discrete(2)})

    def _get_observation(self):
        return {'position': self._state, 'diamond': self._diamond}

    def _4_state_problem(self, action):
        reward = 0
        done = False
        if self._state == 1:  # If we are on the slope
            if action == 0:  # We go left and die!
                reward = -10
                done = True
                self._state = 0
            else:
                # There is a chance that we slip on the slope
                if np.random.random() < 0.05:
                    self._state = 0
                    reward = -10
                    done = True
                else:
                    self._state = 2
                    done = False
                    reward = 0
        elif self._state == 2:  # Initial state
            if action == 0:  # We go left
                if self._diamond:
                    # We get the diamond!
                    reward = 3
                    self._diamond = 0
                self._state = 1
            else:
                # We are done!
                reward = 10
                self._state = 3
                done = True
        return reward, done

    def _5_state_problem(self, action):
        reward = 0
        done = False
        if self._state == 1:  # If we are on the slope
            if action == 0:  # We go left and die!
                reward = -10
                done = True
                self._state = 0
            else:
                # There is a chance that we slip on the slope
                if np.random.random() < 0.05:
                    self._state = 0
                    reward = -10
                    done = True
                else:
                    self._state = 2
                    done = False
                    reward = 0
        elif self._state == 2:  # Initial state
            if action == 0:  # We go left
                if self._diamond:
                    # We get the diamond!
                    reward = 3
                    self._diamond = 0
                self._state = 1
            else:
                if np.random.random() < 0.00:
                    done = True
                    reward = -10
                else:
                    # We are done!
                    if self._diamond_2:
                        self._diamond_2 = 0
                        reward = 3
                    else:
                        reward = 0

                    self._state = 3
                    done = False
        elif self._state == 3:
            if action == 0:
                self._state = 2
            else:
                reward = 10
                self._state = 4
                done = True
        return reward, done

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        reward, done = self.fn(action)
        observation = self._get_observation()

        return observation, reward, done, {}

    def reset(self):
        # The states are:
        #   0: Into the firepit = death
        #   1: On the slope
        #   2: Starting state
        #   3: At the goal
        self._state = 2
        self._diamond = 1
        self._diamond_2 = 1
        return self._get_observation()

    def render(self, mode="human"):
        pass


def init_q_table(shape, method='zeros', seed=42):
    np.random.seed(seed)
    if method == 'zeros':
        q_table = np.zeros(shape=shape)
    elif method == 'ones':
        q_table = np.ones(shape=shape)
    else:  # method == 'uniform':
        q_table = np.random.uniform(low=-1, high=1, size=shape)
    return q_table


def main(problem, episodes, lr, gamma, epsilon):

    env = ToyEnv(problem=problem)

    # Implement a lookup table for the reward
    # There are 4 positions + 2 possibilities for the diamond (there/not there)
    # + 2 actions (left/right) = 4x2x2 = 16 states
    n_positions = env.observation_space['position'].n
    n_diamonds = env.observation_space['diamond'].n
    n_actions = env.action_space.n
    q_table = init_q_table(shape=(n_positions, n_diamonds, n_actions), method='zeros')

    t = tqdm.trange(episodes, desc='Total reward: 0', unit='Episodes', leave=True)
    for episode in t:
        old_state = env.reset()
        done = False
        total_reward = 0
        while not done:
            actions = q_table[(old_state['position'], old_state['diamond'])]
            if np.random.random() > epsilon:
                action = np.argmax(actions)
            else:
                action = np.random.randint(0, env.action_space.n)

            # Step the environment
            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[(new_state['position'], new_state['diamond'])])
            # Current Q value (for current state and performed action)
            current_q = q_table[(old_state['position'], old_state['diamond']) + (action,)]
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - lr) * current_q + lr * (reward + gamma * max_future_q)
            # Update Q table with new Q value
            q_table[(old_state['position'], old_state['diamond']) + (action,)] = new_q

            old_state = new_state
        if episode % 5000 == 0:
            t.set_description(f"Total reward: {total_reward}", refresh=True)

    return q_table


def eval_state(pos, diamond, q_table):
    string_action = {0: 'left', 1: 'right'}
    state = (pos, diamond)
    action = np.argmax(q_table[state])
    print(f'Optimal action in (pos, diamond) = {state}: {string_action[action]}')
    return action


def print_policy(q_table):
    pos = 2
    diamond = 1
    print(f'Printing the optimal policy for the q_table: {q_table}')
    print()
    print(f'Starting in state = (position, diamond) : {pos, diamond}')
    while np.max(q_table[pos, diamond]):
        action = eval_state(pos, diamond, q_table)
        # Move the player
        pos += 1 if action else -1
        # Check if the diamond has been taken
        if diamond and pos == 1:
            diamond = 0


if __name__ == '__main__':
    LEARNING_RATE = 0.0001
    DISCOUNT = 0.9
    EPISODES = 50_000  # Number of episodes to run
    EPSILON = 0.05  # Exploration probability
    PROBLEM = '5state'
    q_table = main(problem=PROBLEM, episodes=EPISODES, lr=LEARNING_RATE, gamma=DISCOUNT, epsilon=EPSILON)
    print_policy(q_table)
