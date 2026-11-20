import torch
from torch import nn
from torch import optim
from agent.network import *

from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminal"))

class Agent:

    def __init__(self, grid_size, num_foods, num_actions, eps, eps_decay, eps_min, gamma, batch_size, memory_capacity, lr):

        self.num_actions = num_actions
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.online_q = Network(grid_size, num_foods, num_actions).to(self.device)
        self.target_q = Network(grid_size, num_foods, num_actions).to(self.device)
        self.q_optim = optim.Adam(self.online_q.parameters(), lr=self.lr, weight_decay = 1e-8)

        self.learn_count = 0
        self.learn_update = 50


    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append(Transition(state, action, reward, next_state, terminal))

    def update_eps(self):
        eps = self.eps * self.eps_decay
        if eps > self.eps_min:
            self.eps = eps

    def select_action(self, state):

        if random.random() > self.eps:
            self.update_eps()
            return random.choice(range(self.num_actions))
        
        self.online_q.eval()

        snake, velocity, foods = state
        snake, velocity, foods = [snake], [velocity], [foods]

        qs = self.online_q(snake, velocity, foods)
        self.update_eps()
        return qs[0].argmax()

    def learn(self):

        sample_size = self.batch_size if self.batch_size < len(self.memory) else len(self.memory)
        batch = random.sample(self.memory, sample_size)

        state_batch = [[], [], []]
        action_batch = []
        reward_batch = []
        next_state_batch = [[], [], []]
        terminal_batch = []

        for sample in batch:

            state = sample.state
            action = sample.action
            reward = sample.reward
            next_state = sample.next_state
            terminal = sample.terminal

            snake, velocity, foods = state
            state_batch[0].append(snake)
            state_batch[1].append(velocity)
            state_batch[2].append(foods)

            action_batch.append(action)
            reward_batch.append(reward)

            snake, velocity, foods = next_state
            next_state_batch[0].append(snake)
            next_state_batch[1].append(velocity)
            next_state_batch[2].append(foods)

            terminal_batch.append(terminal)

        self.online_q.train()
        self.target_q.eval()

        action = torch.tensor(action_batch, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        terminal = torch.tensor(terminal_batch, dtype=torch.bool, device=self.device)

        target_q_next_state = self.target_q(*next_state_batch)

        q_true = reward + self.gamma * target_q_next_state.max(dim=1).values
        q_true[terminal] = reward[terminal]

        online_q_state = self.online_q(*state_batch)
        q_predict = online_q_state[torch.arange(sample_size), action]
        q_error = ((q_true - q_predict) ** 2).mean()

        self.q_optim.zero_grad()

        q_error.backward()
        self.q_optim.step()

        self.learn_count += 1
        if self.learn_count % self.learn_update == 0:
            self.update_target()

        return q_error.item()

    def update_target(self):
        self.target_q.load_state_dict(self.online_q.state_dict())