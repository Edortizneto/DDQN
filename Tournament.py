import gym
from tensorflow import keras
import numpy as np

env = gym.make('LunarLander-v2').env
state = env.reset()
model = keras.models.load_model('data/300model_lunar_land', compile=False)
n_times = 10
DeepQLearning_scores = []
DoubleDeepQLearning_scores = []

print("DQN")
for i in range(n_times):
    done = False
    rewards = 0
    steps = 0

    while not done and steps < 3000:
        Q_values = model.predict(state[np.newaxis])
        action = np.argmax(Q_values[0])
        state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        steps += 1

    DeepQLearning_scores.append(rewards)
    print(f'Score = {rewards}')
    if i < n_times:
        state = env.reset()
        done = False

state = env.reset()
model = keras.models.load_model('data/300model_DD_lunar_land', compile=False)
print("DDQN")
for i in range(n_times):
    done = False
    rewards = 0
    steps = 0

    while not done and steps < 3000:
        Q_values = model.predict(state[np.newaxis])
        action = np.argmax(Q_values[0])
        state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        steps += 1

    DoubleDeepQLearning_scores.append(rewards)
    print(f'Score = {rewards}')
    if i < n_times:
        state = env.reset()
        done = False

D_mean = np.mean(DeepQLearning_scores)
DD_mean = np.mean(DoubleDeepQLearning_scores)

D_std = np.std(DeepQLearning_scores)
DD_std = np.std(DoubleDeepQLearning_scores)

print(f"Deep QLearning \n mean = {D_mean} \t std = {D_std} \
        \nDouble Deep QLearning \n mean = {DD_mean} \t std = {DD_std}")
