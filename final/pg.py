import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torch.distributions.categorical import Categorical

from torch.autograd import Variable

import numpy as np

import gym

from tqdm import tqdm

class Policy(nn.Module):
   def __init__(self, hidden_size=64):
      super(Policy, self).__init__()
      self.affine1 = nn.Linear(8, hidden_size)
      self.affine2 = nn.Linear(hidden_size, 4)

   def forward(self, x):
      x = F.relu(self.affine1(x))
      action_scores = self.affine2(x)
      return F.softmax(action_scores, dim=0)

class Agent:

   def __init__(
      self,
      env=gym.make("LunarLander-v2"),
      learning_rate=3e-4
   ):
      self.env = env
      self.policy = Policy()
      self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

   def select_action(self, state):
      state = Variable(torch.FloatTensor(state))
      action_probs = self.policy(state)
      log_probs = action_probs.log()
      action = Categorical(action_probs).sample()
      return action.data.cpu().numpy(), log_probs[action]

   def play_episode(self):
      state, _ = self.env.reset()

      rewards, log_probs = [], []

      while True:
         action, log_prob = self.select_action(state)
         state, reward, terminated, truncated, info = self.env.step(action)
         
         rewards.append(reward)
         log_probs.append(log_prob)
         
         if terminated or truncated:
            break

      return rewards, log_probs


   def compute_loss(self, rewards, log_probs):
      R = torch.zeros(1, 1).type(torch.FloatTensor)
      loss = 0
      for i in reversed(range(len(rewards))):
         R = R + rewards[i]
         loss = loss - log_probs[i] * Variable(R)
      loss = loss / len(rewards)
      return loss


   def update_policy(self, rewards, log_probs):
      self.optimizer.zero_grad()
      loss = self.compute_loss(rewards, log_probs)
      loss.backward()
      self.optimizer.step()


   def train(self, n_policy_updates):

      for i in tqdm(range(n_policy_updates)):

         # collect trajectories
         rewards, log_probs = self.play_episode()

         self.update_policy(rewards, log_probs)

   
import keyboard

agent = Agent()
agent.train(n_policy_updates=5)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

while True:
#for _ in range(1000):

   if keyboard.is_pressed('esc'):
      break

   action, _ = agent.select_action(observation)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
