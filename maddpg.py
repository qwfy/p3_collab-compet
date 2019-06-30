import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import common
import logging
import os
import functools

logger = logging.getLogger(__name__)

FC_SIZE = 64

def weight_range(layer):
  in_features = layer.weight.data.size()[0]
  rng = 1. / np.sqrt(in_features)
  return -rng, rng


class Actor(nn.Module):

  def __init__(self, state_length, action_length):
    nn.Module.__init__(self)
    self.fc1 = nn.Linear(in_features=state_length, out_features=FC_SIZE)
    self.bn1 = nn.BatchNorm1d(num_features=FC_SIZE)
    self.fc2 = nn.Linear(in_features=FC_SIZE, out_features=FC_SIZE)
    self.bn2 = nn.BatchNorm1d(num_features=FC_SIZE)
    self.fc3 = common.noisy_linear.NoisyLinear(in_features=FC_SIZE, out_features=FC_SIZE)
    self.bn3 = nn.BatchNorm1d(num_features=FC_SIZE)
    self.fc4 = common.noisy_linear.NoisyLinear(in_features=FC_SIZE, out_features=action_length)
    self._init_weights()

  def forward(self, states):
    x = states
    x = F.relu(self.bn1(self.fc1(x)))
    x = F.relu(self.bn2(self.fc2(x)))
    x = F.relu(self.bn3(self.fc3(x)))
    x = F.tanh(self.fc4(x))
    return x

  def sample_epsilon(self):
    with torch.no_grad():
      for x in [self.fc3, self.fc4]:
        x.sample_epsilon()

  def _init_weights(self):
    self.fc1.weight.data.uniform_(*weight_range(self.fc1))
    self.fc2.weight.data.uniform_(*weight_range(self.fc2))


class Critic(nn.Module):

  def __init__(self, state_length, action_length):
    nn.Module.__init__(self)
    self.fc1 = nn.Linear(in_features=state_length, out_features=FC_SIZE)
    self.bn1 = nn.BatchNorm1d(num_features=FC_SIZE)
    self.fc2 = nn.Linear(in_features=FC_SIZE+action_length, out_features=FC_SIZE)
    self.bn2 = nn.BatchNorm1d(num_features=FC_SIZE)
    self.fc3 = nn.Linear(in_features=FC_SIZE, out_features=FC_SIZE)
    self.bn3 = nn.BatchNorm1d(num_features=FC_SIZE)
    self.fc4 = nn.Linear(in_features=FC_SIZE, out_features=1)
    self._init_weights()

  def forward(self, states, actions):
    states = F.relu(self.bn1(self.fc1(states)))
    x = torch.cat([states, actions], dim=1)
    x = F.relu(self.bn2(self.fc2(x)))
    x = F.relu(self.bn3(self.fc3(x)))
    x = self.fc4(x)
    return x

  def _init_weights(self):
    self.fc1.weight.data.uniform_(*weight_range(self.fc1))
    self.fc2.weight.data.uniform_(*weight_range(self.fc2))
    self.fc3.weight.data.uniform_(*weight_range(self.fc3))
    self.fc4.weight.data.uniform_(-3e-3, 3e-3)


class Agent:

  def __init__(self, state_length, action_length, hp, num_homogeneous_agents, num_stacks, writer):
    self._state_length = state_length * num_stacks
    self._action_length = action_length
    self._hp = hp
    self._num_homogeneous_agents = num_homogeneous_agents
    self._writer = writer

    flatten_state_length = self._state_length * self._num_homogeneous_agents
    flatten_action_length = self._action_length * self._num_homogeneous_agents

    self._actors_local = self._map(
      lambda _: Actor(state_length=self._state_length, action_length=self._action_length).cuda())

    self._actors_target = self._map(
      lambda _: Actor(state_length=self._state_length, action_length=self._action_length).cuda())

    self._critics_local = self._map(
      lambda _: Critic(state_length=flatten_state_length, action_length=flatten_action_length).cuda())

    self._critics_target = self._map(
      lambda _: Critic(state_length=flatten_state_length, action_length=flatten_action_length).cuda())

    self._critic_local_optimizers = self._map(
      lambda i: optim.Adam(params=self._critics_local[i].parameters(), lr=self._hp.critic_local_lr))

    self._actor_local_optimizers = self._map(
      lambda i: optim.Adam(params=self._actors_local[i].parameters(), lr=self._hp.actor_local_lr))

    self._map(self._copy_weights)

    logger.info(f'architecture of the actor: %s', self._actors_local[0])
    logger.info(f'architecture of the critic: %s', self._critics_local[0])

    self._memory = common.memory.Uniform(max_size=hp.memory_max_size)

    self._times_learned = 0
    self._experiences_seen = 0
    self._learning_start_reported = False


  def act(self, states):
    states = torch.from_numpy(states.astype(np.float32)).cuda()
    actions = np.zeros((self._num_homogeneous_agents, self._action_length))
    with torch.no_grad():
      for i in range(self._num_homogeneous_agents):
        self._actors_local[i].eval()
        self._actors_local[i].sample_epsilon()
        actions_i = self._actors_local[i](states[i].unsqueeze(0))
        actions[i] = actions_i.cpu().numpy().squeeze(0)
    return actions

  def step(self, states, actions, rewards, next_states, dones, i_episode):
    self._experiences_seen += 1

    experience = common.memory.Experience(
      state=states, action=actions, reward=rewards, next_state=next_states, done=dones)
    self._memory.put(experience)

    if (len(self._memory) >= max(self._hp.batch_size, self._hp.start_learning_memory_size)
      and self._experiences_seen % self._hp.learn_every_new_samples == 0):
      if not self._learning_start_reported:
        logger.info('learning started at episode: %s', i_episode)
        self._learning_start_reported = True
      self._learn()

  def _learn(self):
    self._times_learned += 1

    self._map(lambda i: self._actors_local[i].train())
    self._map(lambda i: self._actors_target[i].train())
    self._map(lambda i: self._critics_local[i].train())
    self._map(lambda i: self._critics_target[i].train())

    with torch.no_grad():
      self._map(lambda i: self._actors_local[i].sample_epsilon())
      self._map(lambda i: self._actors_target[i].sample_epsilon())

    for i_agent in range(self._num_homogeneous_agents):
      self._learn_one_agent(i_agent)

    if self._times_learned % self._hp.update_target_every_learnings == 0:
      self._map(self._soft_update)


  def _learn_one_agent(self, i_agent):
    batch = self._memory.sample(n=self._hp.batch_size)
    batch.states = torch.from_numpy(batch.states).cuda()
    batch.actions = torch.from_numpy(batch.actions).cuda()
    batch.rewards = torch.from_numpy(batch.rewards).cuda()
    batch.next_states = torch.from_numpy(batch.next_states).cuda()
    batch.dones = torch.from_numpy(batch.dones).cuda()

    def f():
      self._map(lambda i: self._actors_local[i].zero_grad())
      self._map(lambda i: self._critics_local[i].zero_grad())

      # calculate the target Q value
      # on the next states, with the target actions on the next states
      with torch.no_grad():
        # each actor only receives its own local observation
        # action of this particular agent on the batch_size next_state,
        # thus the shape should be (batch_size, action_length) (for each agent),
        # and the overall shape should be (batch_size, action_length * num_agents)
        next_as_t = self._map(lambda i: self._actors_target[i](batch.next_states[:, i]))
        next_as_t = torch.cat(next_as_t, dim=1)

        # the i_agent's critic receives all observations, and all actions
        # thus the shape of the input state is (batch_size, state_length * num_agents)
        next_qs_t = self._critics_target[i_agent](self._view_agents_flat(batch.next_states), next_as_t)
        next_qs_t = next_qs_t.squeeze(dim=1)
        qs_t = batch.rewards[:, i_agent] + (1 - batch.dones[:, i_agent]) * self._hp.gamma * next_qs_t

      # train the local critic
      self._critic_local_optimizers[i_agent].zero_grad()
      qs_l = self._critics_local[i_agent](
        self._view_agents_flat(batch.states),
        self._view_agents_flat(batch.actions))
      qs_l = qs_l.squeeze(dim=1)
      critic_loss = F.mse_loss(qs_l, qs_t)
      critic_loss.backward()
      self._critic_local_optimizers[i_agent].step()

      self._writer.add_scalar(f'critic_loss_agent_{i_agent}', critic_loss.item(), self._times_learned)
    f()

    def f():
      self._map(lambda i: self._actors_local[i].zero_grad())
      self._map(lambda i: self._critics_local[i].zero_grad())

      # train the local actor
      as_l = self._map(lambda i: self._actors_local[i](batch.states[:, i]))
      as_l = torch.cat(as_l, dim=1)
      qs_l = self._critics_local[i_agent](self._view_agents_flat(batch.states), as_l)

      policy_loss = -qs_l.mean()
      policy_loss.backward()
      self._actor_local_optimizers[i_agent].step()

      self._writer.add_scalar(f'actor_loss_agent_{i_agent}', policy_loss.item(), self._times_learned)
    f()

  def _soft_update(self, i_agent):
    tau = self._hp.soft_update_tau
    with torch.no_grad():
      for l, t in zip(self._critics_local[i_agent].parameters(), self._critics_target[i_agent].parameters()):
        new_t = tau * l.data + (1 - tau) * t.data
        t.data = new_t
      for l, t in zip(self._actors_local[i_agent].parameters(), self._actors_target[i_agent].parameters()):
        new_t = tau * l.data + (1 - tau) * t.data
        t.data = new_t


  def _copy_weights(self, i_agent):
    with torch.no_grad():
      for l, t in zip(self._critics_local[i_agent].parameters(), self._critics_target[i_agent].parameters()):
        t.data.copy_(l.data)
      for l, t in zip(self._actors_local[i_agent].parameters(), self._actors_target[i_agent].parameters()):
        t.data.copy_(l.data)

  def _save_one(self, directory, i_agent):
    torch.save(self._actors_local[i_agent].state_dict(), os.path.join(directory, f'actor_local_{i_agent}.pt'))
    torch.save(self._actors_target[i_agent].state_dict(), os.path.join(directory, f'actor_target_{i_agent}.pt'))
    torch.save(self._critics_local[i_agent].state_dict(), os.path.join(directory, f'critic_local_{i_agent}.pt'))
    torch.save(self._critics_target[i_agent].state_dict(), os.path.join(directory, f'critic_target_{i_agent}.pt'))

  def save(self, directory):
    self._map(functools.partial(self._save_one, directory))

  def _map(self, f):
    return list(map(f, range(self._num_homogeneous_agents)))

  @staticmethod
  def _view_agents_flat(x):
    return x.view(x.shape[0], -1)
