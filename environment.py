import time
from agent import *

import flappy_bird_gym
env = flappy_bird_gym.make("FlappyBird-v0")

state_space = env.observation_space.shape[0]
agent = DQNAgent()
epochs = 7000
max_score = 100



for i in range(epochs):
    state = env.reset()
    state = np.reshape(state, [1, state_space])
    done = False
    score = 0

    print(f"Epoch: {i}")
    print(f"Epsilon: {agent.epsilon}")

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_space])
        score += 1

        if done:
            reward -= 100
        else:
            reward += 0.5


        # env.render()
        # time.sleep(1/30)

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if score > max_score and i > 10:
            max_score = score
            print(max_score)
            print(f"Yo, this is getting pretty good")
            agent.model.save_weights(f"{i}.hdf5")
        
        if i > 10:
            agent.learn()
    agent.decreaseEpsilon()




env.close()
