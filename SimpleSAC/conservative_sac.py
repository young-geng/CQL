from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

from .model import Scalar
from .sac import soft_target_update


class ConservativeSAC(object):

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
                 target_update_period=1,
                 use_cql=True,
                 cql_n_actions=10,
                 cql_importance_sample=True,
                 cql_lagrange=True,
                 cql_target_action_gap=-1.0,
                 cql_temp=1.0,
                 cql_min_q_weight=5.0,):


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

         # CQL configs
        self.use_cql = use_cql
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_min_q_weight = cql_min_q_weight

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

        if self.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=qf_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch, return_stats=False):
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


        ### CQL
        if not self.use_cql:
            qf_loss = qf1_loss + qf2_loss
        else:
            batch_size = actions.shape[0]
            action_dim = actions.shape[-1]
            cql_random_actions = actions.new_empty((batch_size, self.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            cql_current_actions, cql_current_log_pis = self.policy(observations, repeat=self.cql_n_actions)
            cql_next_actions, cql_next_log_pis = self.policy(next_observations, repeat=self.cql_n_actions)
            cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
            cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

            cql_q1_rand = self.qf1(observations, cql_random_actions)
            cql_q2_rand = self.qf2(observations, cql_random_actions)
            cql_q1_current_actions = self.qf1(observations, cql_current_actions)
            cql_q2_current_actions = self.qf2(observations, cql_current_actions)
            cql_q1_next_actions = self.qf1(observations, cql_next_actions)
            cql_q2_next_actions = self.qf2(observations, cql_next_actions)

            cql_cat_q1 = torch.cat(
                [cql_q1_rand, torch.unsqueeze(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand, torch.unsqueeze(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
            )
            cql_std_q1 = torch.std(cql_cat_q1, dim=1)
            cql_std_q2 = torch.std(cql_cat_q2, dim=1)

            if self.cql_importance_sample:
                random_density = np.log(0.5 ** action_dim)
                cql_cat_q1 = torch.cat(
                    [cql_q1_rand - random_density,
                     cql_q1_next_actions - cql_next_log_pis.detach(),
                     cql_q1_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )
                cql_cat_q2 = torch.cat(
                    [cql_q2_rand - random_density,
                     cql_q2_next_actions - cql_next_log_pis.detach(),
                     cql_q2_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )

            cql_min_qf1_loss = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1).mean() * self.cql_min_q_weight * self.cql_temp
            cql_min_qf2_loss = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1).mean() * self.cql_min_q_weight * self.cql_temp

            """Subtract the log likelihood of data"""
            cql_min_qf1_loss = cql_min_qf1_loss - q1_pred.mean() * self.cql_min_q_weight
            cql_min_qf2_loss = cql_min_qf2_loss - q2_pred.mean() * self.cql_min_q_weight

            if self.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = alpha_prime * (cql_min_qf1_loss - self.cql_target_action_gap)
                cql_min_qf2_loss = alpha_prime * (cql_min_qf2_loss - self.cql_target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                alpha_prime_loss = observations.new_tensor(0.0)
                alpha_prime = observations.new_tensor(0.0)


            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss


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

        if self.use_cql:
            cql_metrics = dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),

                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),

                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),

                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),

                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
            )
        else:
            cql_metrics = {}

        metrics = dict(
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
        metrics.update(cql_metrics)
        return metrics

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
