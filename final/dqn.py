# Deep Q-Learning (Deep Q Network, DQN) with experience replay and epsilon-greedy exploration

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gym
from tqdm import tqdm

import random

import matplotlib.pyplot as plt

class ANN(nn.Module):

   def __init__(self):
      super(ANN, self).__init__()
      self.layer1 = nn.Linear(8, 512)
      self.layer2 = nn.Linear(512, 256)
      self.layer3 = nn.Linear(256, 4)
   
   def forward(self, x):
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      x = self.layer3(x)
      return x


class Agent:

   def __init__(
      self,
      env=gym.make("LunarLander-v2"),
      epsilon=0.1,
      learning_rate=0.001,
      discount_factor=0.99,
      batch_size=32,
      min_epsilon=0.01,
      epsilon_decay=0.995
   ):
      self.Q = ANN()
      self.optimizer = torch.optim.Adam(self.Q.parameters(), learning_rate)
      self.epsilon = epsilon
      self.discount_factor = discount_factor
      self.env = env
      self.buffer = []
      self.batch_size = batch_size
      self.epsilon_decay = epsilon_decay
      self.min_epsilon = min_epsilon

   def select_action(self, state):
      if np.random.random() < self.epsilon:
         return self.env.action_space.sample()

      output = self.Q(torch.FloatTensor(state))
      return torch.argmax(output).item()

   def update_weights(self):
      random_sample = random.sample(self.buffer, self.batch_size)
      states, actions, rewards, next_states, terminated = zip(*random_sample)

      states = torch.FloatTensor(states)
      actions = torch.LongTensor(actions)
      rewards = torch.FloatTensor(rewards)
      next_states = torch.FloatTensor(next_states)
      terminated = torch.FloatTensor(terminated)

      # Q(s, a)
      Q = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

      # max_a Q(s', a)
      max_Q = self.Q(next_states).max(1)[0]

      # target = r + gamma * max_a Q(s', a)
      target = rewards + self.discount_factor * max_Q * (1 - terminated)

      # loss = (Q(s, a) - target)^2
      loss = F.mse_loss(Q, target)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return loss.item()

   def train(self, n_episodes, n_steps=1000):

      episode_rewards = []
      episode_losses = []

      #for episode in tqdm(range(n_episodes)):
      for episode in range(n_episodes):
         state, _ = self.env.reset()

         episode_reward = 0
         losses = []
         
         for step in range(n_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward

            # add to replay memory
            buffer_entry = (
               state.tolist(),
               action,
               reward,
               next_state.tolist(),
               terminated
            )
            self.buffer.append(buffer_entry)

            state = next_state

            if len(self.buffer) > self.batch_size:
               loss = self.update_weights()
               losses.append(loss)

            if terminated or truncated:
               break
         
         if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
         
         if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

         episode_loss = np.mean(losses)

         episode_rewards.append(episode_reward)
         episode_losses.append(episode_loss)

         print(
            f"Episode {episode},",
            f"epsilon {round(self.epsilon, 3)},",
            f"episode reward {round(episode_reward, 3)},",
            f"episode loss {round(episode_loss, 3)}"
         )

      plt.plot(episode_rewards)
      plt.xlabel("Episode")
      plt.ylabel("Reward")
      plt.title("Episode Rewards")
      plt.savefig("dqn_episode_rewards.png")
      plt.close()

      plt.plot(episode_losses)
      plt.xlabel("Episode")
      plt.ylabel("Loss")
      plt.title("Episode Losses")
      plt.savefig("dqn_episode_losses.png")
      plt.close()


import keyboard

agent = Agent()
agent.train(25)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

while True:

   if keyboard.is_pressed('esc'):
      break

   action = agent.select_action(observation)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
