import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import common
import logging
import os

logger = logging.getLogger(__name__)


def weight_range(layer):
  in_features = layer.weight.data.size()[0]
  rng = 1 / np.sqrt(in_features)
  return -rng, rng


class Actor(nn.Module):

  def __init__(self, state_length, action_length):
    nn.Module.__init__(self)
    self.fc1 = nn.Linear(in_features=state_length, out_features=400)
    self.bn1 = nn.BatchNorm1d(num_features=400)
    self.fc2 = nn.Linear(in_features=400, out_features=300)
    self.fc3 = common.noisy_linear.NoisyLinear(in_features=300, out_features=action_length)
    self._init_weights()

  def forward(self, states):
    x = states
    x = F.relu(self.bn1(self.fc1(x)))
    x = F.relu(self.fc2(x))
    x = F.tanh(self.fc3(x))
    return x

  def sample_epsilon(self):
    with torch.no_grad():
      for x in [self.fc3]:
        x.sample_epsilon()

  def _init_weights(self):
    self.fc1.weight.data.uniform_(*weight_range(self.fc1))
    self.fc2.weight.data.uniform_(*weight_range(self.fc2))


class Critic(nn.Module):

  def __init__(self, state_length, action_length):
    nn.Module.__init__(self)
    self.fc1 = nn.Linear(in_features=state_length, out_features=400)
    self.bn1 = nn.BatchNorm1d(num_features=400)
    self.fc2 = nn.Linear(in_features=400 + action_length, out_features=300)
    self.fc3 = nn.Linear(in_features=300, out_features=1)
    self._init_weights()

  def forward(self, states, actions):
    states = F.relu(self.bn1(self.fc1(states)))
    x = torch.cat([states, actions], dim=1)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def _init_weights(self):
    self.fc1.weight.data.uniform_(*weight_range(self.fc1))
    self.fc2.weight.data.uniform_(*weight_range(self.fc2))
    self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class Agent:

  def __init__(self, state_length, action_length, hp, writer):
    self._state_length = state_length
    self._action_length = action_length
    self._hp = hp
    self._writer = writer

    self._actor_local = Actor(state_length=state_length, action_length=action_length).cuda()
    self._actor_target = Actor(state_length=state_length, action_length=action_length).cuda()
    self._critic_local = Critic(state_length=state_length, action_length=action_length).cuda()
    self._critic_target = Critic(state_length=state_length, action_length=action_length).cuda()
    self._copy_weights()

    models = [('actor_local', self._actor_local),
              ('actor_target', self._actor_target),
              ('critic_local', self._critic_local),
              ('critic_target', self._critic_target)]
    for name, model in models:
      logger.info(f'architecture of model %s: %s', name, model)

    self._critic_local_optimizer = optim.Adam(params=self._critic_local.parameters(), lr=self._hp.critic_local_lr)
    self._actor_local_optimizer = optim.Adam(params=self._actor_local.parameters(), lr=self._hp.actor_local_lr)

    self._memory = common.memory.Uniform(max_size=hp.memory_max_size)

    self._times_learned = 0
    self._num_experiences_seen = 0

  def act(self, states):
    states = torch.from_numpy(states.astype(np.float32)).cuda()
    with torch.no_grad():
      self._actor_local.eval()
      self._actor_local.sample_epsilon()
      actions = self._actor_local(states)
      actions = actions.cpu().numpy()
      return actions

  def step(self, states, actions, rewards, next_states, dones):
    self._actor_local.train()
    self._actor_target.train()
    self._critic_local.train()
    self._critic_target.train()

    experiences = zip(states, actions, rewards, next_states, dones)
    for exp in experiences:
      self._memory.put(exp)
      self._num_experiences_seen += 1

      if (len(self._memory) >= max(self._hp.batch_size, self._hp.start_learning_memory_size)
        and self._num_experiences_seen % self._hp.learn_every_new_experiences == 0):
        for _ in range(self._hp.times_consequtive_learn):
          self._learn()

  def _learn(self):
    self._times_learned += 1
    batch = self._memory.sample(n=self._hp.batch_size)
    batch.states = torch.from_numpy(batch.states).cuda()
    batch.actions = torch.from_numpy(batch.actions).cuda()
    batch.rewards = torch.from_numpy(batch.rewards).cuda()
    batch.next_states = torch.from_numpy(batch.next_states).cuda()
    batch.dones = torch.from_numpy(batch.dones).cuda()

    with torch.no_grad():
      self._actor_local.sample_epsilon()
      self._actor_target.sample_epsilon()

    def f():
      # calculate the target Q value
      # on the next states, with the target actions on the next states
      with torch.no_grad():
        next_as_t = self._actor_target(batch.next_states)
        next_qs_t = self._critic_target(batch.next_states, next_as_t)
        next_qs_t = next_qs_t.squeeze(dim=1)
        qs_t = batch.rewards + (1 - batch.dones) * self._hp.gamma * next_qs_t

      # train the local critic
      self._critic_local_optimizer.zero_grad()
      qs_l = self._critic_local(batch.states, batch.actions)
      qs_l = qs_l.squeeze(dim=1)
      critic_loss = F.mse_loss(qs_l, qs_t)
      critic_loss.backward()
      self._critic_local_optimizer.step()

      if self._times_learned % 100 == 0:
        for param in self._critic_local.parameters():
          self._writer.add_histogram('critic_grad', param.grad.cpu().numpy(), self._times_learned)

    f()

    def f():
      # train the local actor
      self._actor_local.zero_grad()
      self._critic_local.zero_grad()
      # [batch, num_actions]
      as_l = self._actor_local(batch.states)
      qs_l = self._critic_local(batch.states, as_l)

      policy_loss = -qs_l.mean()
      policy_loss.backward()
      self._actor_local_optimizer.step()

      if self._times_learned % 100 == 0:
        for param in self._actor_local.parameters():
          self._writer.add_histogram('actor_grad', param.grad.cpu().numpy(), self._times_learned)
        for a in as_l:
          self._writer.add_histogram('actions_local', a.detach().cpu().numpy(), self._times_learned)

    f()

    if self._times_learned % self._hp.update_target_every_learnings == 0:
      self._soft_update()

  def _soft_update(self):
    tau = self._hp.soft_update_tau
    with torch.no_grad():
      for l, t in zip(self._critic_local.parameters(), self._critic_target.parameters()):
        new_t = tau * l.data + (1 - tau) * t.data
        t.data = new_t
      for l, t in zip(self._actor_local.parameters(), self._actor_target.parameters()):
        new_t = tau * l.data + (1 - tau) * t.data
        t.data = new_t

  def _copy_weights(self):
    with torch.no_grad():
      for l, t in zip(self._critic_local.parameters(), self._critic_target.parameters()):
        t.data = l.data
      for l, t in zip(self._actor_local.parameters(), self._actor_target.parameters()):
        t.data = l.data

  def save(self, directory):
    torch.save(self._actor_local.state_dict(), os.path.join(directory, 'actor_local.pt'))
    torch.save(self._actor_target.state_dict(), os.path.join(directory, 'actor_target.pt'))
    torch.save(self._critic_local.state_dict(), os.path.join(directory, 'critic_local.pt'))
    torch.save(self._critic_target.state_dict(), os.path.join(directory, 'critic_target.pt'))
