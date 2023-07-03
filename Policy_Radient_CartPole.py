import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Policy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.fc2 = nn.Linear(128, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = Policy(observation_space, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

    def choose_action(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        probabilities = self.policy(observation)
        action = torch.multinomial(probabilities, 1).item()
        return action

    def calculate_returns(self, rewards, gamma):
        processed_returns = np.zeros_like(rewards)
        sum_returns = 0
        for t in reversed(range(len(rewards))):
            sum_returns = sum_returns * gamma + rewards[t]
            processed_returns[t] = sum_returns
        return processed_returns

    def update_policy(self, observations, actions, rewards, gamma):
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions).view(-1, 1).long()

        probabilities = self.policy(observations)
        log_probabilities = torch.log(probabilities.gather(1, actions))

        processed_returns = self.calculate_returns(rewards, gamma)
        processed_returns = torch.from_numpy(processed_returns).float()
        processed_returns = (processed_returns - torch.mean(processed_returns)) / (
            torch.std(processed_returns) + 1e-9)

        loss = -torch.mean(torch.squeeze(log_probabilities)
                           * processed_returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="human")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)

    num_episodes = 500
    rewards = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_observations = []
        episode_actions = []
        episode_rewards = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            env.render()

            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(
                action)

            episode_observations.append(observation)
            episode_actions.append(action)
            episode_rewards.append(reward)

            observation = next_observation
            episode_reward += reward

        agent.update_policy(
            np.array(episode_observations),
            np.array(episode_actions),
            np.array(episode_rewards),
            gamma=0.95
        )

        rewards.append(episode_reward)
        print(f"Episode: {episode+1}, Reward: {episode_reward}")

    env.close()
