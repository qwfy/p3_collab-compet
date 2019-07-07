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

import time
from unityagents import UnityEnvironment
import maddpg
import tensorboardX
import git
import numpy as np
import collections
import dataclasses
import random
import torch
import itertools
import pickle
import argparse

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
  passes_every_learn: int
  soft_update_tau: float
  save_interval: int
  initial_noise: float
  noise_decay: float


#%%
def setup_env():
  key = 'LD_LIBRARY_PATH'
  so_paths = ['Tennis_Linux_NoVis/Tennis_Data/Plugins/x86_64/',
              'Tennis_Linux_NoVis/Tennis_Data/MonoBleedingEdge/x86_64/']
  so_paths = [os.path.join(os.path.abspath(os.getcwd()), p) for p in so_paths]
  so_paths = ':'.join(so_paths)
  old = os.getenv(key, None)
  if old is not None:
    so_paths = f'{old}:{so_paths}'
  logger.info('%s: %s', key, so_paths)
  os.environ[key] = so_paths


# %%
def train(hp, simulator, unity_worker_id):
  random.seed(1234)
  np.random.seed(2345)
  torch.manual_seed(4567)

  time_start = time.time()
  run_id = time.strftime('%b%d_%H-%M-%S', time.localtime(time_start))
  run_id = f'{run_id}_{unity_worker_id}'

  file_handler = logging.FileHandler(f'run/log/{run_id}.log')
  file_handler.setFormatter(FORMATTER)
  ROOT_LOGGER.addHandler(file_handler)

  repo = git.Repo()

  logger.info('======= run_id %s started at %s =======', run_id, time.strftime('%b%d_%H-%M-%S_%z'))
  logger.info('using commit: %s (clean=%s): %s', repo.head.commit.hexsha, not repo.is_dirty(), repo.head.commit.message)

  logger.info('run_id %s using hyper parameters: %s, unity_worker_id: %s', run_id, hp, unity_worker_id)

  setup_env()

  writer = tensorboardX.SummaryWriter(os.path.join('run/summary', run_id))
  env = UnityEnvironment(file_name=simulator, worker_id=unity_worker_id)
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

  current_noise = hp.initial_noise

  for i_episode in range(hp.num_episodes):
    logger.info('begin episode: %s', i_episode)
    states = env.reset(train_mode=True)[BRAIN_NAME].vector_observations

    # this records the episode reward for each agent
    episode_rewards = np.zeros(NUM_HOMOGENEOUS_AGENTS)

    episode_length = 0

    agent.noise.reset()

    while True:
      current_noise *= hp.noise_decay
      episode_length += 1
      actions = agent.act(states, noise=current_noise)
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

    writer.add_scalar('episode_length', episode_length, i_episode)

    # the episode reward is defined to be the maximum of all agents
    episode_reward = np.max(episode_rewards)
    writer.add_scalar('episode_reward_agent_max', episode_reward, i_episode)
    for i, reward in enumerate(episode_rewards):
      writer.add_scalar(f'agent_{i}/episode_reward', reward, i_episode)

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



def train_partial(arg):
  simulator, unity_worker_id, hp = arg
  return train(hp=hp, simulator=simulator, unity_worker_id=unity_worker_id)

def grid_search():
  many_memory_max_size = [int(1e6)]
  many_memory_initial_alpha = [0.5]
  many_num_episodes = [1500]
  many_batch_size = [64, 256, 1024]
  many_gamma = [0.8, 0.9, 0.95, 0.99]
  many_critic_local_lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  many_actor_local_lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  many_update_target_every_learnings = [1, 3, 10, 20]
  many_learn_every_new_samples = [10, 100, 300]
  many_passes_every_learn = [1, 3, 10, 30, 100]
  many_soft_update_tau = [1e-1, 1e-2, 1e-3]
  many_start_learning_memory_size = [5120]
  many_save_interval = [100]
  many_initial_noise = [0.01, 0.1, 0.3, 0.9]
  many_noise_decay = [0.8, 0.9, 0.99, 0.999, 0.9999]

  cartesian = itertools.product(
    many_memory_max_size,
    many_memory_initial_alpha,
    many_num_episodes,
    many_batch_size,
    many_gamma,
    many_critic_local_lr,
    many_actor_local_lr,
    many_update_target_every_learnings,
    many_learn_every_new_samples,
    many_passes_every_learn,
    many_soft_update_tau,
    many_start_learning_memory_size,
    many_save_interval,
    many_initial_noise,
    many_noise_decay)
  cartesian = [HyperParam(
    memory_max_size=memory_max_size,
    memory_initial_alpha=memory_initial_alpha,
    num_episodes=num_episodes,
    batch_size=batch_size,
    gamma=gamma,
    critic_local_lr=critic_local_lr,
    actor_local_lr=actor_local_lr,
    update_target_every_learnings=update_target_every_learnings,
    learn_every_new_samples=learn_every_new_samples,
    passes_every_learn=passes_every_learn,
    soft_update_tau=soft_update_tau,
    start_learning_memory_size=start_learning_memory_size,
    save_interval=save_interval,
    initial_noise=initial_noise,
    noise_decay=noise_decay,
    ) for (
    memory_max_size,
    memory_initial_alpha,
    num_episodes,
    batch_size,
    gamma,
    critic_local_lr,
    actor_local_lr,
    update_target_every_learnings,
    learn_every_new_samples,
    passes_every_learn,
    soft_update_tau,
    start_learning_memory_size,
    save_interval,
    initial_noise,
    noise_decay) in cartesian
    ]
  random.shuffle(cartesian)

  search_space = list(zip(itertools.repeat('Tennis_Linux_NoVis/Tennis.x86_64'),
                          range(50, 50+len(cartesian)),
                          cartesian))

  logger.info('dumping search space')
  with open('search_space.pkl', 'wb') as f:
    pickle.dump(search_space, f)
  logger.info('dumped search space')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', type=int, required=True)
  args = parser.parse_args()

  with open('search_space.pkl', 'rb') as f:
    search_space = pickle.load(f)
  arg = search_space[args.n]
  del search_space
  train_partial(arg)