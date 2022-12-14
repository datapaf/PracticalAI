import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym

class Policy(nn.Module):
    def __init__(self, discount_factor=0.99):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(8, 128)
        self.layer_a = nn.Linear(128, 4)
        self.layer_v = nn.Linear(128, 1)
        self.discount_factor = discount_factor
        self.log_probs = []
        self.rewards = []
        self.values = []

    def forward(self, x):
        x = F.relu(self.layer1(x))

        probs = F.softmax(self.layer_a(x), dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))

        value = self.layer_v(x)
        self.values.append(value)

        return action.item()

    def compute_loss(self):
        
        # compute discounted rewards
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.discount_factor * discounted_reward
            rewards.insert(0, discounted_reward)

        # normalize discounted rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()

        # compute actor loss
        actor_loss = []
        for log_prob, value, reward in zip(self.log_probs, self.values, rewards):
            advantage = reward - value.item()
            actor_loss.append(-log_prob * advantage)

        # compute critic loss
        critic_loss = []
        for value, reward in zip(self.values, rewards):
            critic_loss.append(F.smooth_l1_loss(value, reward))

        # reset
        self.log_probs = []
        self.rewards = []
        self.values = []

        return torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()

class Agent:

    def __init__(
        self,
        env=gym.make("LunarLander-v2"),
        discount_factor=0.99,
        learning_rate=0.02
    ):
        self.env = env
        self.policy = Policy(discount_factor)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def train(self, n_episodes, n_steps=1000):
        running_reward = 0
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            for step in range(n_steps):
                action = self.policy(torch.FloatTensor(state))
                state, reward, terminated, truncated, info = self.env.step(action)
                self.policy.rewards.append(reward)

                running_reward += reward

                if terminated or truncated:
                    break
            
            self.optimizer.zero_grad()
            loss = self.policy.compute_loss()
            loss.backward()
            self.optimizer.step()

        
            print(
                f"Episode {episode},",
                f"running reward {running_reward:.2f}",
            )
            running_reward = 0

Agent().train(10)
