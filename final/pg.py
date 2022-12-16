# Policy Gradient approach for the LunarLander-v2 environment.
# The expression for the gradient uses log probabilities and rewards to go.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.autograd import Variable

import gym
from tqdm import tqdm

import matplotlib.pyplot as plt

class Policy(nn.Module):
   """
      Policy ANN for the LunarLander-v2 environment powered by PyTorch.
      Input is the state, output is the action probabilities.
   """

   def __init__(self, hidden_size):
      """
         Initialize the policy ANN.
         
         hidden_size: number of hidden units
      """
      super(Policy, self).__init__()
      # create first linear layer with 8 inputs and hidden_size outputs
      self.affine1 = nn.Linear(8, hidden_size)
      # create second linear layer with hidden_size inputs and 4 outputs
      self.affine2 = nn.Linear(hidden_size, 4)

   def forward(self, x):
      """
         Forward pass of the policy ANN.
         
         x: input state
         ouput: action probabilities
      """
      
      # apply ReLU activation function to the output of the first layer
      x = F.relu(self.affine1(x))
      # apply softmax activation function to the output of the second layer
      action_scores = self.affine2(x)
      return F.softmax(action_scores, dim=0)

class Agent:
   """Agent for the LunarLander-v2 environment powered by PyTorch ANN."""

   def __init__(
      self,
      env=gym.make("LunarLander-v2"),
      hidden_size=32,
      learning_rate=3e-4
   ):
      """
         Initialize the agent.
         
         env: environment
         hidden_size: number of hidden units
         learning_rate: learning rate for the optimizer
      """
      self.env = env
      self.policy = Policy(hidden_size)
      self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

   def select_action(self, state):
      """
         Select an action based on the current policy.

         state: current state
         output: action and log probability of the action
      """
      # convert state to a tensor and wrap it in a Variable
      state = Variable(torch.FloatTensor(state))
      # get action probabilities
      action_probs = self.policy(state)
      # get log probabilities
      log_probs = action_probs.log()
      # sample an action from the action probabilities
      action = Categorical(action_probs).sample()

      # return action and log probability of the action
      return action.data.cpu().numpy(), log_probs[action]

   def play_episode(self, n_steps=1000):
      """
         Play one episode and return the rewards and log probabilities.

         n_steps: maximum number of steps in the episode
         output: list of rewards and list of log probabilities
      """

      # reset the environment and get the initial state
      state, _ = self.env.reset()

      # initialize lists for rewards and log probabilities
      rewards, log_probs = [], []

      # play the episode
      for i in range(n_steps):

         # select an action based on the current policy
         action, log_prob = self.select_action(state)
         # take the action and get the next state, reward, and info
         state, reward, terminated, truncated, info = self.env.step(action)
         
         # append reward and log probability to the lists
         rewards.append(reward)
         log_probs.append(log_prob)
         
         # break if the episode is terminated or truncated
         if terminated or truncated:
            break

      return rewards, log_probs


   def compute_loss(self, rewards, log_probs):
      """
         Compute the loss for the policy gradient algorithm.

         rewards: list of rewards
         log_probs: list of log probabilities
         output: loss
      """
      # initialize the reward to go
      R = torch.zeros(1, 1).type(torch.FloatTensor)
      # initialize the loss
      loss = 0
      
      # loop through the rewards in reversed order
      for i in reversed(range(len(rewards))):
         # compute reward to go
         R = R + rewards[i]
         # compute the loss
         loss = loss - log_probs[i] * Variable(R)
      
      # normalize the loss
      loss = loss / len(rewards)
      
      return loss


   def update_policy(self, rewards, log_probs):
      """
         Update the policy based on the rewards and log probabilities.

         rewards: list of rewards
         log_probs: list of log probabilities
         output: loss value
      """
      self.optimizer.zero_grad()
      loss = self.compute_loss(rewards, log_probs)
      loss.backward()
      self.optimizer.step()

      return loss.data.item()


   def train(self, n_episodes):
      """
         Train the agent for n_episodes.

         n_episodes: number of play-episode-update iterations
      """

      episode_rewards = []
      episode_losses = []

      #for i in tqdm(range(n_policy_updates)):
      for i in range(n_episodes):
         rewards, log_probs = self.play_episode()
         loss = self.update_policy(rewards, log_probs)
         episode_reward = sum(rewards)
         episode_rewards.append(episode_reward)
         episode_losses.append(loss)
         
         print(
            f"Episode {i},",
            f"Reward: {episode_reward},",
            f"Loss: {loss}"
         )
      
      plt.plot(episode_rewards)
      plt.xlabel("Episode")
      plt.ylabel("Reward")
      plt.title("Episode Rewards")
      plt.savefig("pg_episode_rewards.png")
      plt.close()

      plt.plot(episode_losses)
      plt.xlabel("Episode")
      plt.ylabel("Loss")
      plt.title("Episode Losses")
      plt.savefig("pg_episode_losses.png")
      plt.close()


# train agent and demonstate it in practice

import keyboard

agent = Agent()
agent.train(1500)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

while True:

   if keyboard.is_pressed('esc'):
      break

   action, _ = agent.select_action(observation)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
