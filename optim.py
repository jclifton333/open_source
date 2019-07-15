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


def gradient_ascent_minmax_parameter(V, V_opponent, params1, params2, player, defect, lr, create_graph=True):
  """
  Take the minmax gradient update if defect.

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


def max_min_exploitability_policy(R_player, R_opponent, player_exploitable, opponent_exploitable):
  """
  If player exploitable, play BR to opponent maxmin
  If opponent exploitable, play maxmin

  :param V:
  :param params1:
  :param params2:
  :param lr:
  :param player:
  :param defect:
  :param R_player:
  :param R_opponent:
  :param reject:
  :return:
  """
  if player_exploitable:  # Play best response to max min
    opponent_maxmin = np.argmax((np.min(R_opponent[:, 0]), np.min(R_opponent[:, 1])))
    return _, np.argmax(R_player[:, opponent_maxmin])
  elif opponent_exploitable:  # Play estimated maxmin
    return _, np.argmax((np.min(R_player[:, 0]), np.min(R_player[:, 1])))


def gradient_ascent_minmax_reward(V, params1, params2, lr, player, defect, R_opponent):
  """
  Take the minmax action with respect to the opponent's estimated reward if defect.

  :param V:
  :param R_opponent:
  :param params1:
  :param params2:
  :param lr:
  :param player:
  :return:
  """
  # Get update
  dVd1, dVd2 = grad(V((params1, params2)), (params1, params2), retain_graph=True, create_graph=True)
  if player == 1:
    update = (dVd1 * lr).data
  else:
    update = (dVd2 * lr).data

  # If defect, return minmax action
  if defect:
    max_a1 = np.max((R_opponent(0, 0), R_opponent(1, 0)))
    max_a2 = np.max((R_opponent(0, 1), R_opponent(1, 1)))
    a_punish = np.argmin((max_a1, max_a2))
  else:
    a_punish = None

  return update, a_punish


