import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad
from random import uniform
from functools import partial
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize
import random
import sys


def constrained_minmax(V, (params1, params2), lr):
  """
   Minmax update of params1 against params2, constrained to have norm less than lr.

  :param V:
  :param lr:
  :return:
  """
  ineq_cons = {'type': 'ineq', 'fun': lambda d: np.norm(d) - lr}

  def outer_problem(delta_param_1):
    def inner_problem(delta_param_2):
      return -V((params1 + delta_param_1, params2 + delta_param_2))
    inner_res = minimize(inner_problem, x0=params2, constraints=[{}, ineq_cons])
    return inner_problem(inner_res.x)

  outer_res = minimize(outer_problem, x0=params1, constraints=[{}, ineq_cons])
  return outer_res.x


def gradient_ascent_minmax(V, (params1, params2), player, defect, lr, create_graph=True):
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
