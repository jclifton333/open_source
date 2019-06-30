import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad
from random import uniform
from functools import partial
from abc import ABCMeta, abstractmethod
import random
import sys


def constrained_minmax(V, (params1, params2), lr):
  pass


def gradient_ascent_minmax(V, (params1, params2), defect, lr, create_graph=True):
  """

  :param V:
  :param defect:
  :param lr:
  :param create_graph:
  :return:
  """
  # ToDo: handle which player this is for

  if defect:
    dVd1 = constrained_minmax(V, (params1, params2), lr)
  else:
    dVd1, _ = grad(V, (params1, params2), create_graph=True)
