import os
import sys
import logging

ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel(logging.INFO)
FORMATTER = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(FORMATTER)
ROOT_LOGGER.addHandler(stream_handler)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.getcwd(), 'python-priorityq'))

os.makedirs('run/model', exist_ok=True)
os.makedirs('run/summary', exist_ok=True)
os.makedirs('run/log', exist_ok=True)

import argparse
import time
from unityagents import UnityEnvironment
import maddpg
import tensorboardX
import git
import numpy as np
import collections
import tqdm
import dataclasses
import random
import torch


# %%
BRAIN_NAME = 'TennisBrain'
NUM_HOMOGENEOUS_AGENTS = 2
SOLVE_NUM_EPISODES = 100
SOLVE_REWARD = 0.5
NUM_STACKS = 3


# %%
@dataclasses.dataclass
class HyperParam:
  memory_max_size: int
  memory_initial_alpha: float
  start_learning_memory_size: int
  num_episodes: int
  batch_size: int
  gamma: float
  critic_local_lr: float
  actor_local_lr: float
  update_target_every_learnings: int
  learn_every_new_samples: int
  soft_update_tau: float
  save_interval: int


# %%
def train(hp, cli_args):
  random.seed(1234)
  np.random.seed(2345)
  torch.manual_seed(4567)

  time_start = time.time()
  run_id = time.strftime('%b%d_%H-%M-%S_%z', time.localtime(time_start))

  file_handler = logging.FileHandler(f'run/log/{run_id}.log')
  file_handler.setFormatter(FORMATTER)
  ROOT_LOGGER.addHandler(file_handler)

  repo = git.Repo()

  logger.info('======= run_id %s started at %s =======', run_id, time.strftime('%b%d_%H-%M-%S_%z'))
  logger.info('using commit: %s (clean=%s): %s', repo.head.commit.hexsha, not repo.is_dirty(), repo.head.commit.message)

  logger.info('run_id %s using hyper parameters %s', run_id, hp)

  writer = tensorboardX.SummaryWriter(os.path.join('run/summary', run_id))
  pbar = tqdm.tqdm(total=hp.num_episodes)

  env = UnityEnvironment(file_name=cli_args.simulator, worker_id=cli_args.unity_worker_id)
  state_length = env.brains[BRAIN_NAME].vector_observation_space_size
  action_length = env.brains[BRAIN_NAME].vector_action_space_size

  agent = maddpg.Agent(
    state_length=state_length,
    action_length=action_length,
    hp=hp,
    num_homogeneous_agents=NUM_HOMOGENEOUS_AGENTS,
    num_stacks=NUM_STACKS,
    writer=writer)

  window_rewards = collections.deque(maxlen=SOLVE_NUM_EPISODES)

  last_save_episode = None

  for i_episode in range(hp.num_episodes):
    states = env.reset(train_mode=True)[BRAIN_NAME].vector_observations

    # this records the episode reward for each agent
    episode_rewards = np.zeros(NUM_HOMOGENEOUS_AGENTS)

    episode_length = 0

    while True:
      episode_length += 1
      actions = agent.act(states)
      env_info = env.step(actions)[BRAIN_NAME]
      next_states = env_info.vector_observations
      rewards = env_info.rewards
      dones = env_info.local_done

      agent.step(states, actions, rewards, next_states, dones, i_episode)
      episode_rewards += rewards

      if any(dones):
        break
      else:
        states = next_states

    pbar.update(1)

    writer.add_scalar('episode_length', episode_length, i_episode)

    # the episode reward is defined to be the maximum of all agents
    episode_reward = np.max(episode_rewards)
    writer.add_scalar('episode_reward_agent_max', episode_reward, i_episode)
    for i, reward in enumerate(episode_rewards):
      writer.add_scalar(f'episode_reward_agent_{i}', reward, i_episode)

    window_rewards.append(episode_reward)
    mean_reward = np.mean(window_rewards)
    writer.add_scalar(f'episode_reward_agent_max_avg_over_{SOLVE_NUM_EPISODES}_episodes', mean_reward, i_episode)

    if (len(window_rewards) >= SOLVE_NUM_EPISODES
      and mean_reward >= SOLVE_REWARD
      and (last_save_episode is None
           or i_episode - last_save_episode >= hp.save_interval)):

      last_save_episode = i_episode
      save_dir = os.path.join('run/model', f'{run_id}_{i_episode}')
      os.makedirs(save_dir, exist_ok=True)
      agent.save(save_dir)
      logger.info('model saved to directory: %s', save_dir)

  time_stop = time.time()
  logger.info(
    'run_id %s completed at %s, time cost: %s seconds',
    run_id,
    time.strftime('%b%d_%H-%M-%S_%z', time.localtime(time_stop)),
    f'{time_stop - time_start:.2f}')
  env.close()


# %%
HP = HyperParam(
  memory_max_size=int(1e6),
  memory_initial_alpha=0.5,
  num_episodes=30000,
  batch_size=1024,
  gamma=0.95,
  critic_local_lr=1e-3,
  actor_local_lr=1e-3,
  update_target_every_learnings=5,
  learn_every_new_samples=128,
  soft_update_tau=1e-3,
  start_learning_memory_size=10240,
  save_interval=100)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--simulator', type=str, default='Tennis_Linux_NoVis/Tennis.x86_64')
  parser.add_argument('--unity_worker_id', type=int, default=0)
  args = parser.parse_args()
  train(hp=HP, cli_args=args)