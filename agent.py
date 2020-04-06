import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.9            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.9        # L2 weight decay
TRAIN_EVERY = 20

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.steps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.step_window = 1
        self.state_list = deque(maxlen=self.step_window)
        self.state_act_list = deque(maxlen=self.step_window)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) #, weight_decay=WEIGHT_DECAY)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done, update_target=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        #Enqueue state in list
        self.state_list.append(state)
        if len(self.state_list) != self.step_window : ##Only if I have enough steps in the window
            return
        
        #This function receives states, actions, etc. of one agent at a time
        # Save experience / reward
        #[3, 20, 33]
        #sts = np.asarray(self.state_list) #reshaping 3 states to single list.
        #sts = sts.reshape(20, self.step_window * self.state_size)
        
        self.memory.add(list(self.state_list), action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.steps += 1
        if len(self.memory) > BATCH_SIZE and (self.steps % TRAIN_EVERY) == 0 :
            for _ in range(10) :
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, update_target)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        #If not enough states to act with the NN, then return random action (maybe same last action?)
        #state has the last state of 20 agents
        self.state_act_list.append(state)
        if len(self.state_act_list) != self.step_window :
            return np.clip(np.random.rand(20, self.action_size), -1, 1)

        #Convert the state array to tensor
        #state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        
        with torch.no_grad():
            st = np.asarray([e for el in self.state_act_list for e in el], dtype=np.float32)
            st = torch.from_numpy(st).float().unsqueeze(0).to(device)
            st = torch.reshape(st, (20, self.step_window * self.state_size)) #dense input [20, 3*33]
            action_values = self.actor_local(st).cpu().data.numpy()
            
            #Input the state to the actor's local network (the regular) to get the "best" action
#             action = self.actor_local(state).cpu().data.numpy()
        
        #Perform a training step
        self.actor_local.train()
        #Add noise to the obtained action
        if add_noise:
            action_values += self.noise.sample()
        
        #Ensure the output is in the valid action's range [-1, 1]
        return np.clip(action_values, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, update_target=False):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
                
        states, actions, rewards, next_states, dones = experiences

        # states = torch.reshape(states, (BATCH_SIZE, self.state_size * self.step_window))
        # next_states = torch.cat((states, next_states), 1)[:,self.state_size:]
        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Get action from actor's network, given the NEXT state        
        actions_next = self.actor_target(next_states)
        # Use the obtained action as input to the critic's network, along with the NEXT state
        Q_targets_next = self.critic_target(next_states, actions_next)
        ###Q_targets_next = self.qnetwork_target(new_st).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        #Get the expected value from the critic's local network
        Q_expected = self.critic_local(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        #Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        #Get the "best" action using the actor's local network, given current state
        actions_pred = self.actor_local(states)
        
        #Compute the loss by getting the expected value (V) from the critic's local network,
        #given the current state and the obtained action. 
        #Set it negative to perform gradient ascent?
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if update_target :
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.1, sigma=0.08):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)