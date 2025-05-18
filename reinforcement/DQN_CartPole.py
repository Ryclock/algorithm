import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random

input_size = 4
hidden_size = 128
output_size = 2
buffer_capacity = 10000
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.001
num_episodes = 1000
max_timesteps = 500


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated, truncated = zip(*batch)
        return (
            torch.stack([torch.from_numpy(s) for s in state]),
            torch.LongTensor(action),
            torch.Tensor(reward),
            torch.stack([torch.from_numpy(s) for s in next_state]),
            torch.Tensor(terminated),
            torch.Tensor(truncated)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, buffer_capacity, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_model = DQN(input_size, hidden_size,
                                output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def select_action(self, state):
        if torch.rand(1) > self.epsilon:
            with torch.no_grad():
                state = torch.Tensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state)
                action = q_values.argmax().item()
        else:
            action = torch.randint(0, self.output_size, (1,)).item()
        return action

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, terminateds, truncateds = self.buffer.sample(
            self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminateds = terminateds.to(self.device)
        truncateds = truncateds.to(self.device)

        current_q_values = self.model(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * \
            next_q_values * (1 - terminateds-truncateds)

        loss = self.loss_fn(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def train(self, num_episodes, max_timesteps):
        env = gym.make('CartPole-v1')
        total_rewards = []

        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0

            for _ in range(max_timesteps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(
                    action)
                episode_reward += reward

                self.buffer.push(
                    (state, action, reward, next_state, terminated, truncated))
                state = next_state

                self.update_model()
                self.decay_epsilon()

                if terminated or truncated:
                    break

            total_rewards.append(episode_reward)
            self.update_target_model()

            print(
                f"Episode {episode}/{num_episodes}, Reward: {episode_reward}")

        return total_rewards


if __name__ == "__main__":
    agent = DQNAgent(input_size, hidden_size, output_size, buffer_capacity,
                     batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay)
    rewards = agent.train(num_episodes, max_timesteps)

    print("Training finished.")
    print(f"Average Reward: {sum(rewards) / num_episodes}")
