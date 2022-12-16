# Actor-Critic algorithm for LunarLander-v2
# The formula for the loss uses the advantage and smooth L1 between V and reward.
# The simple change in V does not work well, so I used difference between V and reward. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym

import matplotlib.pyplot as plt

class Policy(nn.Module):
    """Policy ANN for the LunarLander-v2 environment powered by PyTorch."""

    def __init__(self, discount_factor=0.99):
        """
            Initialize the policy ANN with input layer and two output layers for 
            action probabilities and state values.

            discount_factor: discount factor for the rewards
        """
        super(Policy, self).__init__()
        # create input layer
        self.layer1 = nn.Linear(8, 128)
        # create output layer for action probabilities
        self.layer_a = nn.Linear(128, 4)
        # create output layer for state values
        self.layer_v = nn.Linear(128, 1)
        # discount factor for the rewards
        self.discount_factor = discount_factor

        # lists for log probabilities, rewards and state values
        self.log_probs = []
        self.rewards = []
        self.values = []

    def forward(self, x):
        """
            Forward pass of the policy ANN. While passing the model saves the state values in the list.

            x: input state
            ouput: action
        """
        # apply ReLU activation function to the output of the first layer
        x = F.relu(self.layer1(x))

        # save log probabilities
        probs = F.softmax(self.layer_a(x), dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))

        # save state values
        value = self.layer_v(x)
        self.values.append(value)

        return action.item()

    def compute_loss(self):
        """
            Compute the loss for the policy gradient algorithm.
            The formula for the loss uses the advantage and smooth L1 between V and reward.

            output: loss
        """

        # compute discounted rewards
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.discount_factor * discounted_reward
            rewards.insert(0, discounted_reward)

        # normalize discounted rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()

        # compute loss
        loss = 0
        for log_prob, value, reward in zip(self.log_probs, self.values, rewards):
            advantage = reward - value.item()
            loss = loss - log_prob * advantage + F.smooth_l1_loss(value, torch.tensor([reward]))

        # reset
        self.log_probs = []
        self.rewards = []
        self.values = []

        return loss


class Agent:
    """Agent for the LunarLander-v2 environment."""

    def __init__(
        self,
        env=gym.make("LunarLander-v2"),
        discount_factor=0.99,
        learning_rate=0.02
    ):
        """Initialize the agent with the environment, policy and optimizer."""
        self.env = env
        self.policy = Policy(discount_factor)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
            Select an action based on the state.

            state: current state
            output: action
        """
        return self.policy(torch.FloatTensor(state))

    def train(self, n_episodes, n_steps=1000):
        """
            Train the agent for n_episodes each with maximum n_steps.

            n_episodes: number of episodes
            n_steps: maximum number of steps per episode
        """
        
        # lists for episode rewards and losses
        episode_rewards = []
        episode_losses = []

        # train for n_episodes
        for episode in range(n_episodes):

            # reset environment
            state, _ = self.env.reset()
            
            # reset episode reward
            episode_reward = 0
            
            # play episode for n_steps
            for step in range(n_steps):
                # select action
                action = self.select_action(state)
                # perform action and get new state, reward, termination and truncation
                state, reward, terminated, truncated, info = self.env.step(action)
                # save reward
                self.policy.rewards.append(reward)

                # update episode reward
                episode_reward += reward

                # stop episode if terminated or truncated
                if terminated or truncated:
                    break
            
            # update policy weights
            self.optimizer.zero_grad()
            loss = self.policy.compute_loss()
            loss.backward()
            self.optimizer.step()

            # save episode reward and loss
            episode_rewards.append(episode_reward)
            episode_losses.append(loss.item())
        
            # print episode reward and loss
            print(
                f"Episode {episode},",
                f"episode reward {episode_reward:.3f}",
                f"episode loss {loss.item():.3f}"
            )

        # plot episode rewards and losses

        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        plt.savefig("ac_episode_rewards.png")
        plt.close()

        plt.plot(episode_losses)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Episode Losses")
        plt.savefig("ac_episode_losses.png")
        plt.close()

# train agent and demonstate it in practice

import keyboard

agent = Agent()
agent.train(750)

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

