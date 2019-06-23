import os
import sys
import logging
import socket

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
file_handler = logging.FileHandler(f'run/run.{socket.gethostname()}.log')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.getcwd(), 'python-priorityq'))

os.makedirs('run/model', exist_ok=True)
os.makedirs('run/summary', exist_ok=True)

import time
from unityagents import UnityEnvironment
import maddpg
import tensorboardX
import git
import numpy as np
import collections
import tqdm
import dataclasses


# %%
SIMULATOR_PATH = 'Tennis_Linux/Tennis.x86_64'
BRAIN_NAME = 'TennisBrain'
NUM_AGENTS = 20
SOLVE_NUM_EPISODES = 100
SOLVE_REWARD = 30.0


# %%
@dataclasses.dataclass
class HyperParam:
  memory_max_size: int
  num_episodes: int
  batch_size: int
  gamma: float
  critic_local_lr: float
  actor_local_lr: float
  update_target_every: int
  learn_every_steps: int
  learn_passes: int
  soft_update_tau: float
  ou_theta: float
  ou_sigma: float


# %%
def train(hp):
  time_start = time.time()
  run_id = time.strftime('%b%d_%H-%M-%S_%z', time.localtime(time_start))
  repo = git.Repo()

  logger.info('======= run_id %s started at %s =======', run_id, time.strftime('%b%d_%H-%M-%S_%z'))
  logger.info('using commit: %s (clean=%s): %s', repo.head.commit.hexsha, not repo.is_dirty(), repo.head.commit.message)

  logger.info('run_id %s using hyper parameters %s', run_id, hp)

  writer = tensorboardX.SummaryWriter(os.path.join('run/summary', run_id))
  pbar = tqdm.tqdm(total=hp.num_episodes)

  env = UnityEnvironment(file_name=SIMULATOR_PATH)
  state_length = env.brains[BRAIN_NAME].vector_observation_space_size
  action_length = env.brains[BRAIN_NAME].vector_action_space_size

  agent = maddpg.Agent(
    state_length=state_length,
    action_length=action_length,
    hp=hp,
    num_agents=NUM_AGENTS,
    writer=writer)

  window_rewards = collections.deque(maxlen=SOLVE_NUM_EPISODES)

  for i_episode in range(hp.num_episodes):
    states = env.reset(train_mode=True)[BRAIN_NAME].vector_observations
    episode_reward = 0

    while True:
      actions = agent.act(states)
      env_info = env.step(actions)[BRAIN_NAME]
      next_states = env_info.vector_observations
      rewards = env_info.rewards
      dones = env_info.local_done

      agent.step(states, actions, rewards, next_states, dones)
      episode_reward += np.mean(rewards)

      if any(dones):
        break
      else:
        states = next_states

    pbar.update(1)
    window_rewards.append(episode_reward)
    writer.add_scalar('episode_reward', episode_reward, i_episode)
    mean_reward = np.mean(window_rewards)
    writer.add_scalar(f'episode_reward_avg_over_{SOLVE_NUM_EPISODES}_episodes', mean_reward, i_episode)
    if len(window_rewards) >= SOLVE_NUM_EPISODES and mean_reward >= SOLVE_REWARD:
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
# [paper]: Continuous Control with Deep Reinforcement Learning
HP = HyperParam(
  memory_max_size=int(1e6),  # [paper]
  num_episodes=300,
  batch_size=128,
  gamma=0.99,  # [paper]
  critic_local_lr=1e-3,  # [paper]
  actor_local_lr=1e-3,
  update_target_every=1,
  learn_every_steps=20,
  learn_passes=10,
  soft_update_tau=1e-3,  # [paper]
  ou_theta=0.15,  # [paper]
  ou_sigma=0.2)  # [paper]


if __name__ == '__main__':
  train(hp=HP)