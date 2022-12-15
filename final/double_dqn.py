# Deep Q-Learning (Deep Q Network, DQN) with experience replay and epsilon-greedy exploration.

# The creative addition of the implementation is the use of epslion decay and double DQN approach.
# The epsilon decay is used to reduce the exploration rate over time.
# The double DQN approach is used to reduce the overestimation of the Q-values.
# The main difference between the Vanilla DQN and the Double DQN is the target equation, which is as follows:
# target = r + gamma * Q(s', argmax_a Q(s', a)) instead of target = r + gamma * max_a Q(s', a).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gym
from tqdm import tqdm

import random

import matplotlib.pyplot as plt

class ANN(nn.Module):
   """Neural Network for the Q-function approximation"""

   def __init__(self):
      """Initialize the neural network and its layers"""
      super(ANN, self).__init__()
      
      # Input layer that takes the 8 values of the state as input 
      self.layer1 = nn.Linear(8, 512)
      
      # Hidden layer with 512 neurons
      self.layer2 = nn.Linear(512, 256)
      
      # Output layer with 4 neurons, one for each action
      self.layer3 = nn.Linear(256, 4)
   
   def forward(self, x):
      """
         Forward pass of the neural network
         
         x: state tensor with 8 values
         output: Q-values for each action
      """
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      x = self.layer3(x)
      return x


class Agent:
   """Agent that interacts with the environment"""

   def __init__(
      self,
      env=gym.make("LunarLander-v2"),
      epsilon=1.0,
      learning_rate=0.001,
      discount_factor=0.99,
      batch_size=32,
      min_epsilon=0.01,
      epsilon_decay=0.995
   ):
      """
         Initialize the agent and its parameters

         env: environment to interact with
         epsilon: exploration rate
         learning_rate: learning rate for the optimizer
         discount_factor: discount factor for the Bellman equation
         batch_size: batch size for the experience replay
         min_epsilon: minimum exploration rate
         epsilon_decay: decay rate for the exploration rate
      """
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
      """
         Select an action based on the epsilon-greedy exploration strategy.

         state: current state of the environment
         output: action to take
      """

      # take a random action with probability epsilon
      if np.random.random() < self.epsilon:
         return self.env.action_space.sample()

      # otherwise, take the action with the highest Q-value
      output = self.Q(torch.FloatTensor(state))
      return torch.argmax(output).item()

   def update_weights(self):
      """
         Update the weights of the neural network using the experience replay.

         output: value of the loss
      """

      # sample a batch of transitions from the replay buffer
      random_sample = random.sample(self.buffer, self.batch_size)
      states, actions, rewards, next_states, terminated = zip(*random_sample)

      # convert the batch to tensors
      states = torch.FloatTensor(states)
      actions = torch.LongTensor(actions)
      rewards = torch.FloatTensor(rewards)
      next_states = torch.FloatTensor(next_states)
      terminated = torch.FloatTensor(terminated)

      # Q(s, a)
      Q = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

      # max_a Q(s', a)
      #max_Q = self.Q(next_states).max(1)[0]

      # vanilla dqn target
      # # target = r + gamma * max_a Q(s', a)
      # target = rewards + self.discount_factor * max_Q * (1 - terminated)

      # argmax_a Q(s', a)
      argmax_Q = torch.argmax(self.Q(next_states), 1).unsqueeze(1)

      # Q(s', argmax_a Q(s', a))
      Q_next = self.Q(next_states).gather(1, argmax_Q).squeeze(1)

      # double dqn target
      # target = r + gamma * Q(s', argmax_a Q(s', a))
      target = rewards + self.discount_factor * Q_next * (1 - terminated)

      # loss = (Q(s, a) - target)^2
      loss = F.mse_loss(Q, target)

      # update the weights of the neural network
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return loss.item()

   def train(self, n_episodes, n_steps=1000):
      """
         Train the agent on several episodes each with a maximum of n_steps.
         The method also saves the figures of the rewards and losses.

         n_episodes: number of episodes to train on
         n_steps: maximum number of steps per episode
      """

      # lists to save the rewards and losses
      episode_rewards = []
      episode_losses = []

      # train the agent on n_episodes
      for episode in range(n_episodes):

         # reset the environment
         state, _ = self.env.reset()

         # reset the episode reward and losses
         episode_reward = 0
         losses = []
         
         # run the episode for n_steps
         for step in range(n_steps):

            # select an action
            action = self.select_action(state)
            # take the action and observe the next state and reward
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # add the reward to the episode reward
            episode_reward += reward

            # add to replay memory (s, a, r, s', t)
            buffer_entry = (
               state.tolist(),
               action,
               reward,
               next_state.tolist(),
               terminated
            )
            self.buffer.append(buffer_entry)

            # update the state
            state = next_state

            # update the weights of the neural network if the replay buffer is full enough
            if len(self.buffer) > self.batch_size:
               loss = self.update_weights()
               losses.append(loss)

            # break the episode if terminated or truncated
            if terminated or truncated:
               break
         
         # decay the exploration rate
         if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
         
         # make sure the exploration rate is not below the minimum
         if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

         # save the episode reward and loss
         episode_loss = np.mean(losses)
         episode_rewards.append(episode_reward)
         episode_losses.append(episode_loss)

         # print the episode information
         print(
            f"Episode {episode},",
            f"epsilon {round(self.epsilon, 3)},",
            f"episode reward {round(episode_reward, 3)},",
            f"episode loss {round(episode_loss, 3)}"
         )
      
      # save the figures of the rewards and losses
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


# train agent and demonstate it in practice

import keyboard

agent = Agent()
agent.train(600)

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
