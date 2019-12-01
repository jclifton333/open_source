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


def vanilla_gradient(V, params1, params2, lr):
  dVd1, dVd2 = grad(V((params1, params2)), (params1, params2), retain_graph=True, create_graph=True)
  return (dVd1 * lr).data, (dVd2 * lr).data


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


def max_min_exploitability_policy(R_player_1, R_player_2, player_1_exploitable, player_2_exploitable):
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
  if player_1_exploitable:
    R_exploitable = R_player_1
    R_exploiter = R_player_2
  elif player_2_exploitable:
    R_exploitable = R_player_2
    R_exploiter = R_player_1

  # Get actions
  exploiter_maxmin = np.argmax((np.min(R_exploiter[:, 0]), np.min(R_exploiter[:, 1])))
  exploitable_best_response = np.argmax((R_exploitable[:, exploiter_maxmin]))
  ipw = 1.  # Play is deterministic here so ipw = 1.

  # Assign actions to player number
  a1 = player_1_exploitable*exploitable_best_response + player_2_exploitable*exploiter_maxmin
  a2 = player_2_exploitable*exploitable_best_response + player_1_exploitable*exploiter_maxmin

  return a1, a2, ipw


def gradient_ascent_minmax_reward(V, V1, V2, params1, params2, lr, player, defect, R_opponent):
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


def lola(V, V1, V2, params1, params2, lr, player, defect, R_opponent):
  dV1d1, dV1d2 = grad(V1((params1, params2)), (params1, params2), create_graph=True)
  dV2d1, dV2d2 = grad(V2((params1, params2)), (params1, params2), create_graph=True)
  if player == 1:
    dV2d21 = T.stack([grad(d, params1, retain_graph=True)[0] for d in dV2d2])
    # ToDo: assuming learning rates are equal.
    update = lr * lr * T.matmul(dV1d2, dV2d21).data
  else:
    dV1d12 = T.stack([grad(d, params2, retain_graph=True)[0] for d in dV1d1])
    update = lr * lr * T.matmul(dV2d1, dV1d12).data
  return update, None


def naive_gradient_ascent(V, V2, params1, params2, lr, player, defect, R_opponent):
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
  dVd1, dVd2 = grad(V2((params1, params2)), (params1, params2), retain_graph=True, create_graph=True)
  if player == 1:
    update = (dVd1 * lr).data
  else:
    update = (dVd2 * lr).data

  a_punish = None
  return update, a_punish

