import collections
import random
import numpy as np
import priorityq
from typing import List


Experience = collections.namedtuple(
  'Experience',
  field_names=['state', 'action', 'reward', 'next_state', 'done'])


def unzip(xss):
  if len(xss) == 0:
    raise ValueError(f'Cannot unzip an empty list')
  else:
    tuple_size = len(xss[0])
    result = tuple([[] for _ in range(tuple_size)])
    for xs in xss:
      for i in range(tuple_size):
        result[i].append(xs[i])
    return result


class Batch:
  def __init__(self, experiences: List[Experience]):
    # [(s, a, r, s, d)]
    self.states, self.actions, self.rewards, self.next_states, self.dones = unzip(experiences)
    self.states = np.array(self.states).astype(np.float32)
    self.actions = np.array(self.actions).astype(np.float32)
    self.rewards = np.array(self.rewards).astype(np.float32)
    self.next_states = np.array(self.next_states).astype(np.float32)
    self.dones = np.array(self.dones).astype(np.float32)


class ScoredExperience:
  def __init__(self, score, experience):
    self.score = score
    self.experience = experience

  def __lt__(self, other):
    return self.score < other.score

  def __gt__(self, other):
    return self.score > other.score

class ScoredBatch(Batch):
  def __init__(self, scored_experiences: List[ScoredExperience]):
    super().__init__([scored.experience for scored in scored_experiences])
    self.scored_experiences = scored_experiences

class Uniform:

  def __init__(self, max_size):
    self._q = collections.deque(maxlen=max_size)

  def sample(self, n):
    return Batch(random.choices(self._q, k=n))

  def put(self, exp):
    self._q.append(exp)

  def __len__(self):
    return len(self._q)


class RankPrioritized:

  def __init__(self, max_size):
    self._max_size = max_size
    self._q = priorityq.MappedQueue()

  def sample(self, n, alpha):
    total_length = len(self._q)
    ps = [1 / (total_length - i) for i in range(total_length)]
    pas = np.power(ps, alpha)
    Ps = pas / np.sum(pas)

    return ScoredBatch(random.choices(self._q.h, weights=Ps, k=n))

  def put(self, scored):
    if scored.score is None:
      # first time we see this example
      scored.score = float('inf')
      if len(self._q) >= self._max_size:
        self._q.pop()
      self._q.push(scored)
    else:
      self._q.update(scored, scored)

  def __len__(self):
    return len(self._q)
