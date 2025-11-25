import numpy as np
import random
import torch
from ray import tune
import ray

from AuGraph_env import AuGraphEnv
from AuGraph_model import AuGraphModel
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.models.catalog import ModelCatalog

seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

# 运行ray
ray.shutdown()
ray.init()
ModelCatalog.register_custom_model('augraph_model', AuGraphModel)  # 使用自定义模型
tunerun = tune.run(
    DDPGTrainer,
    # 填写ddpg的相关参数，可以搜索某些参数，比如学习率，奖励gamma等
    config={
        # 其他
        'env': AuGraphEnv,
        'framework': 'torch',
        'seed': seed_num,
        # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # GPU
        'num_gpus': 0,  # GPU，需要<1

        # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
        # TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
        # In addition to settings below, you can use "exploration_noise_type" and
        # "exploration_gauss_act_noise" to get IID Gaussian exploration noise
        # instead of OU exploration noise.
        # twin Q-net
        "twin_q": True,
        # delayed policy update
        "policy_delay": 1,
        # target policy smoothing
        # (this also replaces OU exploration noise with IID Gaussian exploration
        # noise, for now)
        "smooth_target_policy": True,
        # gaussian stddev of target action noise for smoothing
        "target_noise": 0.2,
        # target noise limit (bound)
        "target_noise_clip": 0.5,


        # === Evaluation ===
        # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        # Note that evaluation is currently not parallelized, and that for Ape-X
        # metrics are already only reported for the lowest epsilon workers.
        "evaluation_interval": None,
        # Number of episodes to run per evaluation period.
        "evaluation_duration": 10,

        # ========= Model ============
        # 在进入actor和critic的隐藏层之前，会先运行'model'里的参数
        "use_state_preprocessor": True,  # 可以使用自定义model
        "actor_hiddens": [128, 64],
        "actor_hidden_activation": "relu",
        # Postprocess the critic network model output with these hidden layers;
        # again, if use_state_preprocessor is True, then the state will be
        # preprocessed by the model specified with the "model" config option first.
        "critic_hiddens": [128, 64],
        "critic_hidden_activation": "relu",
        "n_step": 1,  # N-step Q learning
        # 自定义模型
        'model': {
            'custom_model': 'augraph_model',
            'conv_filters': [[36, [3, 3], 1], [18, [2, 2], 1], [6, [2, 2], 1]],
            'conv_activation': 'relu',  # tune.grid_search(['relu','tanh']),
            "post_fcnet_hiddens": [256],
            "post_fcnet_activation": 'relu',  # tune.grid_search(['relu','tanh'])
        },

        # === Exploration ===
        "explore": True,
        "exploration_config": {
            # TD3 uses simple Gaussian noise on top of deterministic NN-output
            # actions (after a possible pure random phase of n timesteps).
            "type": "GaussianNoise",
            # For how many timesteps should we return completely random actions,
            # before we start adding (scaled) noise?
            "random_timesteps": 5000,
            # "random_timesteps": 10000,
            # Gaussian stddev of action noise for exploration.
            "stddev": 0.1,  # tune.grid_search([0.1, 0.15]),
            # Scaling settings by which the Gaussian noise is scaled before
            # being added to the actions. NOTE: The scale timesteps start only
            # after(!) any random steps have been finished.
            # By default, do not anneal over time (fixed 1.0).
            "initial_scale": 1.0,
            "final_scale": 1.0,
            "scale_timesteps": 1,
        },
        # Number of env steps to optimize for before returning
        'timesteps_per_iteration': 100,  # 每次迭代step数量
        # Extra configuration that disables exploration.
        "evaluation_config": {
            "explore": False
        },

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 20000,
        "replay_buffer_config": {
            "type": "MultiAgentReplayBuffer",
            "capacity": 50000,
        },
        # Set this to True, if you want the contents of your buffer(s) to be
        # stored in any saved checkpoints as well.
        # Warnings will be created if:
        # - This is True AND restoring from a checkpoint that contains no buffer
        #   data.
        # - This is False AND restoring from a checkpoint that does contain
        #   buffer data.
        "store_buffer_in_checkpoints": False,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.7,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.3,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-4,
        # Whether to LZ4 compress observations
        "compress_observations": False,

        # The intensity with which to update the model (vs collecting samples from
        # the env). If None, uses the "natural" value of:
        # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
        # `num_envs_per_worker`).
        # If provided, will make sure that the ratio between ts inserted into and
        # sampled from the buffer matches the given value.
        # Example:
        #   training_intensity=1000.0
        #   train_batch_size=250 rollout_fragment_length=1
        #   num_workers=1 (or 0) num_envs_per_worker=1
        #   -> natural value = 250 / 1 = 250.0
        #   -> will make sure that replay+train op will be executed 4x as
        #      often as rollout+insert op (4 * 250 = 1000).
        # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further details.
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for the critic (Q-function) optimizer.
        "critic_lr": 1e-4,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 1e-4,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 2000,
        # Update the target by \tau * policy + (1-\tau) * target_policy
        "tau": 0.001,
        # If True, use huber loss instead of squared loss for critic network
        # Conventionally, no need to clip gradients if using a huber loss
        "use_huber": False,
        # Threshold of a huber loss
        "huber_threshold": 1.0,
        # Weights for L2 regularization
        "l2_reg": 1e-6,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1500,
        # Update the replay buffer with this many samples at once. Note that this
        # setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": 10,
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 256,
        'gamma': 0.98,     # 奖励衰减

        # === Parallelism ===
        # Number of workers for collecting samples with. This only makes sense
        # to increase if your environment is particularly slow to sample, or if
        # you're using the Async or Ape-X optimizers.
        "num_workers": 0,
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent reporting frequency from going lower than this time span.
        "min_time_s_per_reporting": 1,
        # Experimental flag.
        # If True, the execution plan API will not be used. Instead,
        # a Trainer's `training_iteration` method will be called as-is each
        # training iteration.
        "_disable_execution_plan_api": False,
    },
    checkpoint_at_end=True,  # 结束时存储检查点
    checkpoint_freq=1,      # 检查点之间的训练迭代次数
    # 隔几个training_iteration存储一次
    # restore=path #载入检查点
    stop={
        'training_iteration': 200  # 训练轮次
    }
)


# 保存一个最大的训练好的agent
best_checkpoint = tunerun.get_best_checkpoint(
    trial=tunerun.get_best_logdir('episode_reward_mean', 'max'),
    metric='episode_reward_mean',
    mode='max'
)
print(best_checkpoint)
