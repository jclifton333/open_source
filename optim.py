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
import pdb


def constrained_minmax(V_opponent, params1, params2, lr, player):
  """
   Minmax update of params1 against params2, constrained to have norm less than lr.

  :param V_opponent:
  :param lr:
  :return:
  """
  params1 = params1.detach().numpy()
  params2 = params2.detach().numpy()
  ineq_cons = {'type': 'ineq', 'fun': lambda d: np.linalg.norm(d) - lr}

  # ToDo: Currently unconstrained
  def outer_problem(delta_param_i):
    def inner_problem(delta_param_j):
      if player == 1:
        return -V_opponent(params1 + delta_param_i, params2 + delta_param_j)
      else:
        return -V_opponent(params1 + delta_param_j, params2 + delta_param_i)
    inner_res = minimize(inner_problem, x0=np.zeros(5))
    return inner_problem(inner_res.x)

  outer_res = minimize(outer_problem, x0=np.zeros(5))
  return outer_res.x


def gradient_ascent_minmax(V, V_opponent, params1, params2, player, defect, lr, create_graph=True):
  """

  :param V:
  :param defect:
  :param lr:
  :param create_graph:
  :return:
  """
  if defect:
    update = constrained_minmax(V_opponent, params1, params2, lr, player)
    update = T.from_numpy(update).float()
  else:
    dVd1, dVd2 = grad(V((params1, params2)), (params1, params2), retain_graph=True, create_graph=True)
    if player == 1:
      update = (dVd1 * lr).data
    else:
      update = (dVd2 * lr).data
  return update

