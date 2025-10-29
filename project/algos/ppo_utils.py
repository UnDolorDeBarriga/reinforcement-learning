"""
Policy Network - PPO Assignment (Blank Template)
------------------------------------------------
This file defines the skeleton of the PPO actor-critic policy.

Your tasks:
1. Implement the forward pass for both actor and critic networks.
2. Construct the action distribution using Normal and Independent distributions.
3. Understand how the log standard deviation controls exploration.

Fill in code sections marked with:
    # ===== YOUR CODE STARTS HERE =====
    # ===== YOUR CODE ENDS HERE =====

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        # ===== YOUR CODE STARTS HERE =====
        # Define actor (policy) and critic (value) networks using torch.nn.Linear layers.
        # Example: self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        # You should create two separate networks:
        #   - Actor: outputs mean action (μ)
        #   - Critic: outputs state value V(s)
        self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_a = torch.nn.Linear(hidden_size, action_space)

        # Critic network: maps state → hidden → scalar V(s)
        self.fc1_c = torch.nn.Linear(state_space, hidden_size)
        self.fc2_c = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_c = torch.nn.Linear(hidden_size, 1)
        # ===== YOUR CODE ENDS HERE =====

        # ===== YOUR CODE STARTS HERE =====
        # Initialize self.actor_logstd using the environment’s action range:
        # stds = (env.action_space.high - env.action_space.low)
        # self.actor_logstd = np.log(stds)
        # Also store a copy (self.actor_logstd_dist) for later scaling.
        stds = (env.action_space.high - env.action_space.low)
        logstd_np = np.log(stds)

        # store as torch tensors
        self.actor_logstd = torch.tensor(logstd_np, dtype=torch.float32)
        self.actor_logstd_dist = self.actor_logstd.clone()
        # ===== YOUR CODE ENDS HERE =====

    def set_logstd_ratio(self, ratio):
        """Adjust exploration by scaling the actor's log standard deviation."""
        # ===== YOUR CODE STARTS HERE =====
        # Scale self.actor_logstd by ratio (e.g., self.actor_logstd = self.actor_logstd_dist * ratio)
        self.actor_logstd = self.actor_logstd_dist * ratio
        # ===== YOUR CODE ENDS HERE =====

    def init_weights(self):
        """Initialize network weights."""
        # ===== YOUR CODE STARTS HERE =====
        # Iterate over self.modules() and initialize torch.nn.Linear layers:
        #   torch.nn.init.normal_(m.weight, 0, 1e-1)
        #   torch.nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                torch.nn.init.zeros_(m.bias)
        # ===== YOUR CODE ENDS HERE =====

    def forward(self, x):
        """
        Forward pass through actor and critic networks.

        Inputs:
        - x: state tensor

        Outputs:
        - action_dist: a torch.distributions.Independent(Normal) distribution over actions
        - x_c: critic's scalar value estimate
        """
        # ===== YOUR CODE STARTS HERE =====
        # 1. Compute the actor forward pass and output the mean action (μ).
        # 2. Compute the critic forward pass and output state value V(s).
        # 3. Compute the action’s log standard deviation (expand to match μ shape).
        # 4. Compute action standard deviation via torch.exp.
        # 5. Create and return an Independent(Normal(mean, std), 1) distribution and the critic value.
        x_a = F.relu(self.fc1_a(x))
        x_a = F.relu(self.fc2_a(x_a))
        mu = self.fc3_a(x_a)

        # Critic: compute value V(s)
        x_c = F.relu(self.fc1_c(x))
        x_c = F.relu(self.fc2_c(x_c))
        v = self.fc3_c(x_c)

        # Expand logstd to same shape as mu
        log_std = self.actor_logstd.expand_as(mu)
        std = torch.exp(log_std)

        # Create a multivariate (diagonal) Normal distribution
        dist = Independent(Normal(mu, std), 1)

        return dist, v