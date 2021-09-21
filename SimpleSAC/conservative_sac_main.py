import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch
import d4rl

import absl.app
import absl.flags

from .conservative_sac import ConservativeSAC
from .replay_buffer import ReplayBuffer, batch_to_torch, get_d4rl_dataset
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v0',
    max_traj_length=1000,
    replay_buffer_size=1000000,
    seed=42,
    device='cpu',

    policy_arch='256-256',
    qf_arch='256-256',
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    n_train_step_per_epoch=1000,
    eval_period=20,
    eval_n_trajs=5,

    batch_size=256,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)

    dataset_replay_buffer = ReplayBuffer(
        max_size=0, data=get_d4rl_dataset(eval_sampler.env)
    )

    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
    )

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        FLAGS.qf_arch
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        FLAGS.qf_arch
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    metrics = {}
    for epoch in range(FLAGS.n_epochs):

        metrics['epoch'] = epoch

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = dataset_replay_buffer.sample(FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(sac.train(batch), 'sac')
                    )
                else:
                    sac.train(batch)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wandb_logger.log(metrics)


if __name__ == '__main__':
    absl.app.run(main)
