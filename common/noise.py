import numpy as np

class OrnsteinUhlenbeck:

  def __init__(self, shape, theta, sigma):
    self._shape = shape
    self._theta = theta
    self._sigma = sigma

    self._x = None
    self.reset()

  def sample(self):
    dx = - self._theta * self._x + self._sigma * np.random.random(size=self._shape)
    self._x += dx
    return self._x

  def reset(self):
    self._x = np.zeros(self._shape)