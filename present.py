import time
from agent import *

import flappy_bird_gym
env = flappy_bird_gym.make("FlappyBird-v0")

state_space = env.observation_space.shape[0]
agent = DQNAgent()
agent.epsilon = 0

state = env.reset()
state = np.reshape(state, [1, state_space])
done = False
score = 0


while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    score += 1

    env.render()
    time.sleep(1/30)

    state = next_state



env.close()