import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

if len(sys.argv) != 2:
    print('Usage: car.py Continuous for Continuous task and car.py Discrete for Discrete task.')
    exit(0)

Environment = sys.argv[1]
print("You are solving ", Environment, " task :)")

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, legal_actions):
        self.legal_actions = legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def get_value(self, state):
        value = max(self.get_qvalue(state, action) for action in self.legal_actions)
        return value

    def update(self, state, action, reward, next_state):
        """
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * (V(s') - Q(s, a))
        """
        gamma = self.discount
        learning_rate = self.alpha

        qvalue = (1 - learning_rate) * self.get_qvalue(state, action) + \
                 learning_rate * (reward + gamma * (self.get_value(next_state) - self.get_qvalue(state, action)))

        self.set_qvalue(state, action, qvalue)

    def get_best_action(self, state):
        qvalues = np.array([self.get_qvalue(state, action) for action in self.legal_actions])
        best_action = self.legal_actions[qvalues.argmax()]
        return best_action

    def get_action(self, state):
        epsilon = self.epsilon

        explore_now = random.uniform(0, 1)
        if explore_now < epsilon:
            chosen_action = np.random.choice(self.legal_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

def change(x, n_digit_pos=2, n_digit_vel=2):
    # np.array -> tuple to become hashable, round to become Discrete
    return tuple([np.round(x[0], n_digit_pos), np.round(x[1], n_digit_vel)])

def play_and_train(env, agent):
    total_reward = 0.0
    obs = env.reset()
    state = change(obs)
    ground = obs[1]
    done = False
    while not done:
        a = agent.get_action(state)
        if Environment == 'Continuous':
            next_state, reward, done, _ = env.step([a])
        else:
            next_state, reward, done, _ = env.step(a)
        modified_reward = reward + 100 * (abs(next_state[1]) - abs(ground))
        next_state = change(next_state)
        agent.update(state, a, modified_reward, next_state)

        state = next_state
        total_reward += reward
    #print('steps = ', steps, 'reward = ', total_reward)
    return total_reward

if Environment == 'Continuous':
  env = gym.make("MountainCarContinuous-v0")
  legal_actions = np.arange(-1, 1.2, 0.01)
  num_iter = 301
  alpha = 0.5
else:
  env = gym.make("MountainCar-v0").env
  legal_actions = [0, 2]
  num_iter = 701
  alpha = 0.35

env.reset()
agent = QLearningAgent(alpha=alpha, epsilon=0.1, discount=1,
                    legal_actions = legal_actions)
rewards = []
for i in range(num_iter):
    rewards.append(play_and_train(env, agent))
    agent.epsilon *= 0.99
    if i % 100 == 0 and i > 0:
        print('number of iterations ', i, 'last 15 rewards ', np.mean(rewards[-15:]))

plt.plot(rewards[-100:])
plt.ylabel('rewards')
plt.xlabel('epoch')
#plt.show()
plt.savefig('Continuous_last100.png')

plt.plot(rewards)
plt.ylabel('rewards')
plt.xlabel('epoch')
#plt.show()
plt.savefig('Continuous.png')