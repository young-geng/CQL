from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

from .model import Scalar


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


class SAC(object):

    def __init__(self,
                 policy, qf1, qf2, target_qf1, target_qf2,
                 discount=0.99,
                 reward_scale=1.0,
                 alpha_multiplier=1.0,
                 use_automatic_entropy_tuning=True,
                 target_entropy=-3e-2,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 optimizer_type='adam',
                 soft_target_update_rate=5e-3,
                 target_update_period=1):


        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.discount = discount
        self.reward_scale = reward_scale
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.target_entropy = target_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.optimizer_type = optimizer_type
        self.soft_target_update_rate = soft_target_update_rate
        self.target_update_period = target_update_period

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), qf_lr
        )

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=policy_lr,
            )
        else:
            self.log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch):
        self._total_steps += 1

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        new_actions, log_pi = self.policy(observations)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)

        """ Policy loss """
        q_new_actions = torch.min(
            self.qf1(observations, new_actions),
            self.qf2(observations, new_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        new_next_actions, next_log_pi = self.policy(next_observations)
        target_q_values = torch.min(
            self.target_qf1(next_observations, new_next_actions),
            self.target_qf2(next_observations, new_next_actions),
        ) - alpha * next_log_pi

        q_target = self.reward_scale * rewards + (1. - dones) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, q_target.detach())
        qf2_loss = F.mse_loss(q2_pred, q_target.detach())
        qf_loss = qf1_loss + qf2_loss

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        if self.total_steps % self.target_update_period == 0:
            self.update_target_network(
                self.soft_target_update_rate
            )

        return dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        if self.use_automatic_entropy_tuning:
            return (
                self.policy, self.qf1, self.qf2, self.target_qf1,
                self.target_qf2, self.log_alpha
            )
        return (
            self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2
        )

    @property
    def total_steps(self):
        return self._total_steps
