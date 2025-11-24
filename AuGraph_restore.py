import numpy as np
import random
import torch
from ray import tune
import ray
import os
from ray.rllib.agents.ddpg import DDPGTrainer
from AuGraph_env import AuGraphEnv
from AuGraph_env_restore import AuGraphEnvRestore
from ray.rllib.models.catalog import ModelCatalog
from AuGraph_model import AuGraphModel
import Compute
import Database
import AuOdlConvert
import AuGraph
import Restore_path
import Service

## 设置随机种子
seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

## 运行ray
ray.shutdown()
ray.init()
ModelCatalog.register_custom_model('augraph_model', AuGraphModel)  # 使用自定义模型

# 参数配置，把ddpg文件的粘过来就可以
config_re = {
    # 其他
    'env': AuGraphEnv,
    'framework': 'torch',
    'seed': seed_num,
    # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # GPU
    'num_gpus': 0,  # GPU，需要<1

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

    # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
    "twin_q": True,  # twin Q-net
    "policy_delay": 1,  # delayed policy update，1-4都可以
    "smooth_target_policy": True,  # target policy smoothing
    'target_noise': 0.1,  # gaussian stddev of target action noise for smoothing
    "target_noise_clip": 0.3,  # target noise limit (bound),不超过0.5

    # === Evaluation ===
    "evaluation_interval": None,
    "evaluation_num_episodes": 10,  # Number of episodes to run per evaluation period.

    # === Exploration ===
    "explore": True,
    "exploration_config": {
        # TD3 uses simple Gaussian noise on top of deterministic NN-output
        # actions (after a possible pure random phase of n timesteps).
        "type": "GaussianNoise",
        # For how many timesteps should we return completely random actions,
        # before we start adding (scaled) noise?
        "random_timesteps": 0,
        # Gaussian stddev of action noise for exploration.
        "stddev": 0.05,  # tune.grid_search([0.02,0.03,0.04,0.05]), #0.15,0.2不太好
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
    'buffer_size': 50000,
    # If True prioitized replay buffer will be used.
    "prioritized_replay": True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Time steps over which the beta parameter is annealed.
    'prioritized_replay_beta_annealing_timesteps': 20000,
    # Final value of beta
    "final_prioritized_replay_beta": 0.3,
    # Epsilon to add to the TD errors when updating priorities.
    'prioritized_replay_eps': 1e-4,
    # Whether to LZ4 compress observations
    "compress_observations": False,
    # If set, this will fix the ratio of replayed from a buffer and learned on
    # timesteps to sampled from an environment and stored in the replay buffe4
    # timesteps. Otherwise, the replay will proceed at the native ratio
    # determined by (train_batch_size / rollout_fragment_length).
    "training_intensity": None,

    # ========= Optimization ==============
    # cl是al的0.1-1倍
    # Learning rate for the critic (Q-function) optimizer.
    'critic_lr': 1e-4,  # tune.grid_search([1e-4, 6e-5]), #1e-4,cl变成1e-5之后不收敛
    # Learning rate for the actor (policy) optimizer.
    'actor_lr': 1e-5,  # tune.grid_search([1e-5,2e-5]),
    # Update the target network every `target_network_update_freq` steps.
    'target_network_update_freq': 0,  # 2000
    # Update the target by \tau * policy + (1-\tau) * target_policy
    'tau': 0.001,  # 软更新系数略小于1,和知乎里面刚好的相反的
    # If True, use huber loss instead of squared loss for critic network
    # Conventionally, no need to clip gradients if using a huber loss
    "use_huber": False,  # no Huber loss
    # Threshold of a huber loss
    "huber_threshold": 1.0,
    # Weights for L2 regularization,TD3 no l2 regularisation
    "l2_reg": 1e-6,  # 1e-6,
    # If not None, clip gradients during optimization at this value
    "grad_clip": None,
    "clip_rewards": False,
    # How many steps of the model to sample before learning starts.
    'learning_starts': 20000,  # tune.grid_search([10000,5000]), # tune.choice([1500,2000,2500]),
    # Update the replay buffer with this many samples at once.
    "rollout_fragment_length": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    'train_batch_size': 128,
    'gamma': 0.98,  # tune.grid_search([0.98, 0.99]),      # 奖励衰减

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you're using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,
    # "num_gpus_per_worker": 0,

}

path = Restore_path.path

# 恢复经过训练的agent
agent = DDPGTrainer(config=config_re, env=AuGraphEnvRestore)
agent.restore(path)
env = AuGraphEnv({})
episode_reward = 0
done = False
obs = env.reset()

count = 0  # 可以在测试的时候也多跑几次，取一个最好的，把explore开启之后是能选到训练时最好的路径
reward_max = -10000
reward_list = []    # 奖励集合
while count < 20:
    virtual_hop_cumulate = 0  # 累计虚拟拓扑跳数
    virtual_hop_cumulate_list = []  # 统计结果
    while not done:
        action = agent.compute_action(obs)
        # action = agent.compute_action(obs,explore=False)
        obs, reward, done, info = env.step(action)
        # print("Action:", action*1000)
        # print("State:", obs)
        # print("Reward:", reward)
        print('------')
        episode_reward += reward
        print("Total Reward:", episode_reward)
        # virtual_hop_cumulate += virtual_hop_num
        # virtual_hop_cumulate_list.append(virtual_hop_num)
    # if episode_reward >= reward_max:  # 只把最好结果的路径记录下来
    #     reward_max = episode_reward
    #     # print('reward_max', reward_max)
    #     AuOdlConvert.odl_result(AuGraphEnv.au_edge_list)
    #
    #     # print("RWA", AuOdlConvert.result_rwa_phy)
    #     path_count = 0
    #     path_length = []
    #     while path_count < len(AuOdlConvert.result_rwa_phy):
    #         length = int((len(AuOdlConvert.result_rwa_phy[path_count]) - 1) / 2) + int(
    #             (len(AuOdlConvert.result_rwa_phy[path_count + 1]) - 1) / 2)
    #         path_length.append(length)
    #         # path_length.append(int((len(AuOdlConvert.result_rwa_phy[path_count])-1)/2) + int((len(AuOdlConvert.result_rwa_phy[path_count+1])-1)/2))
    #         path_count += 2
    #     print("路径长度")
    #     print("ADMIRE")
    #     for pathi in path_length:
    #         print(pathi)
    #
    #     print("累计虚拟拓扑跳数")
    #     print(virtual_hop_cumulate)
    #     print("虚拟拓扑跳数50")
    #     print("ADMIRE")
    #     index = 1
    #     while index < len(virtual_hop_cumulate_list):
    #         virtual_hop_cumulate_list[index] += virtual_hop_cumulate_list[index - 1]
    #         index += 2
    #
    #     index = 1
    #     while index < len(virtual_hop_cumulate_list):
    #         print(virtual_hop_cumulate_list[index])
    #         index += 2
        # print(Service.path1)

        # with open('rwa_phy.txt', 'w') as f:
        #     for i in range(len(AuOdlConvert.result_rwa_phy)):
        #         f.write(str(AuOdlConvert.result_rwa_phy[i]))
        #         f.write('\n')
        #     f.close()
        #
        # with open('rwa_vir.txt', 'w') as file:
        #     for i in range(len(AuOdlConvert.result_rwa_vir)):
        #         file.write(str(AuOdlConvert.result_rwa_vir[i]))
        #         file.write('\n')
        #     file.close()
        #
        # # 写物理链路
        # with open('phyLinks-GA.txt', 'w') as file:
        #     for data in Database.links_physical:
        #         np.savetxt(file, data, fmt='%.3f', delimiter='\t')
        #         # file.write('\n')
        #     file.close()
        #
        # with open('link_vir.txt', 'w') as file:
        #     for k in range(len(AuGraph.links_virtual_list)):
        #         file.write(str(AuGraph.links_virtual_list[k]))
        #         file.write('\n')
        #     file.close()
        #
        # with open('odl_iog.txt', 'w') as file:
        #     for i in range(len(AuOdlConvert.result_odl)):
        #         file.write(str(AuOdlConvert.result_odl[i]))
        #         file.write('\n')
        #     file.close()

    reward_list.append(episode_reward/50)
    count += 1
    episode_reward = 0
    obs = env.reset()
    done = False


print("奖励列表",reward_list)
# print("Total Reward:", episode_reward)
# print("使用波长", Compute.compute(Database.links_physical))
