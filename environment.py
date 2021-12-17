import time
from agent import *

import flappy_bird_gym
env = flappy_bird_gym.make("FlappyBird-v0")

state_space = env.observation_space.shape[0]
agent = DQNAgent()
epochs = 1000

for i in range(epochs):
    state = env.reset()
    state = np.reshape(state, [1, state_space])
    done = False
    score = 0

    print(f"Epoch: {i}")


    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        score += 1

        if done:
            reward -= 100
        #
        # env.render()
        # time.sleep(1/30)

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if score >= 1000:
            print(f"Yo, this is getting pretty good")
            agent.model.save_weights(f"{score}.hdf5")

        agent.learn()





env.close()