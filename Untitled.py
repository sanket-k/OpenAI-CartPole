
# coding: utf-8

# In[ ]:


import gym
import numpy as np


# In[ ]:


#env = gym.make('CartPole-v0')
#env.reset()


# In[ ]:


def rome_around():
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())


# In[ ]:


#rome_around()


# In[ ]:


def primitive():
    env = gym.make('CartPole-v0')
    for i_episode in range(10):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print('episode finished after {} timesteps'.format(t+1))
                break


# In[ ]:


#primitive()


# In[ ]:


def primitive_2():
    env = gym.make('MountainCar-v0')
    for i_episode in range(10):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print('episode finished after {} timesteps'.format(t+1))
                break


# In[ ]:


#primitive_2()


# In[ ]:


def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    #for 200 timesteps
    for _ in range(200):
        env.render()
        #initialize random weights
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


# In[ ]:


def train(submit):
    env = gym.make('CartPole-v0')
    episode_per_update = 5
    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2 - 1
    best_reward = 0
    
    #2000 episodes
    for _ in range(2000):
        new_params = parameters + (np.random.rand(4) * 2 - 1 ) * noise_scaling
        reward = run_episode(env, parameters)
        print('reward = {0} best = {1}'.format(reward,best_reward))
        if reward > best_reward:
            best_reward = reward
            parameters = new_params
            if reward == 200:
                break


# In[ ]:


r = train(submit=False)
print(r)


# In[ ]:


print(np.random.rand(4))


# In[ ]:




