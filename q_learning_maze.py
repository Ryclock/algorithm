import numpy as np
import sys
import time
from tqdm import tqdm


class QLearningMazeSolver:
    def __init__(self, maze, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
        self.maze = maze
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.explore_rate = epsilon
        self.num_episodes = num_episodes

        self.n_rows, self.n_cols = maze.shape
        self.n_actions = 4  # up,donw,left,right
        self.Q_table = np.zeros((self.n_rows, self.n_cols, self.n_actions))
        self.agent_start_state = self.find_start_state()

        self.solve_path = []

    def find_start_state(self):
        row, col = np.where(self.maze == 3)
        if len(row) == 0:
            raise ValueError("No start state found in the maze.")
        return (row[0], col[0])

    def choose_action(self, agent_state):
        if np.random.uniform(0, 1) < self.explore_rate or np.all(self.Q_table[agent_state] == 0):
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q_table[agent_state])

    def take_action(self, agent_state, action):
        row, col = agent_state
        if action == 0:
            return row - 1, col
        elif action == 1:
            return row + 1, col
        elif action == 2:
            return row, col - 1
        elif action == 3:
            return row, col + 1

    def is_out_of_bounds(self, agent_state):
        row, col = agent_state
        return row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols

    def is_valid_point(self, agent_state):
        return not self.is_out_of_bounds(agent_state) and self.maze[agent_state] != 1

    def update_Q_tables(self, agent_state, action, agent_next_state, reward):
        q_value = self.Q_table[agent_state + (action,)]
        max_q_value = np.max(self.Q_table[agent_next_state])
        new_q_value = (1 - self.learning_rate) * q_value + self.learning_rate * \
            (reward + self.discount_factor * max_q_value - q_value)

        self.Q_table[agent_state + (action,)] = new_q_value

    def train(self):
        for _ in tqdm(range(self.num_episodes), desc="Training"):
            agent_state = self.agent_start_state
            while True:
                action = self.choose_action(agent_state)
                agent_next_state = self.take_action(agent_state, action)
                if not self.is_valid_point(agent_next_state):
                    agent_next_state = agent_state
                    reward = -1
                elif self.maze[agent_next_state] == 2:
                    reward = 1
                else:
                    reward = 0

                self.update_Q_tables(agent_state, action,
                                     agent_next_state, reward)
                agent_state = agent_next_state
                # self.print_maze_state(agent_state)

                if self.maze[agent_state] == 2:
                    break

                # sys.stdout.write("\033[F"*self.n_rows)
                # sys.stdout.flush()

    def solve(self):
        agent_state = self.agent_start_state
        while True:
            action = self.choose_action(agent_state)
            agent_next_state = self.take_action(agent_state, action)
            if not self.is_valid_point(agent_next_state):
                agent_next_state = agent_state

            agent_state = agent_next_state
            self.solve_path.append(agent_state)
            self.print_maze_state(agent_state)

            if self.maze[agent_state] == 2:
                break

            sys.stdout.write("\033[F"*self.n_rows)
            sys.stdout.flush()
            time.sleep(0.1)

    def print_maze_state(self, agent_state):
        maze_state = ""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if (i, j) == agent_state:
                    maze_state += "+ "
                elif self.maze[i, j] == 0:
                    maze_state += "□ "
                elif self.maze[i, j] == 1:
                    maze_state += "■ "
                elif self.maze[i, j] == 2:
                    maze_state += "★ "
                elif self.maze[i, j] == 3:
                    maze_state += "◉ "
            maze_state += "\n"
        sys.stdout.write(maze_state)

    def print_solve_path(self):
        print(f"\n路径长度：{len(self.solve_path)}")
        for pos in self.solve_path:
            print(pos)


if __name__ == "__main__":
    maze = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [3, 0, 0, 0, 0],
    ])
    try:
        solver = QLearningMazeSolver(maze)
        solver.train()
        solver.solve()
        solver.print_solve_path()
    except ValueError as e:
        print(e)
        sys.exit(1)
