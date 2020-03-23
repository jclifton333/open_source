import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad, set_detect_anomaly
from random import uniform
from functools import partial
from scipy.special import expit, logit
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
import pdb
import optim
import random
import sys
from scipy.optimize import minimize

set_detect_anomaly(True)
"""
2x2 games. Players model each other as Boltzmann-rational with variable temperature parameter. 
"""


def onehot(length, index):
  encoding = np.zeros(length)
  encoding[index] = 1
  return encoding


def optimal_profile(payoffs):
  best_payoff = -float('inf')
  best_profile = None
  for a1 in range(payoffs.shape[0]):
    for a2 in range(payoffs.shape[1]):
      if payoffs[a1, a2] > best_payoff:
        best_payoff = payoffs[a1, a2]
        best_profile = (a1, a2)
  return best_profile


def choice_likelihood(probs_and_choices, payoffs_mi_prior, temp=1., n_interactions=5):
  def log_lik(payoffs):
    total_payoff = 0.
    for probs1, probs2, choice in probs_and_choices:  # Compute myopic total
      total_payoff += np.dot(probs1, payoffs)*(1-choice) + np.dot(probs2, payoffs)*choice
    myopic_payoffs_estimate = maximize_myopic_likelihood(probs_and_choices, temp=temp)  # Myopic estimate of my reward
    estimated_welfare_optimal_profile = optimal_profile(myopic_payoffs_estimate + payoffs_mi_prior)
    expected_interaction_payoffs = n_interactions * payoffs.reshape((2, 2))[estimated_welfare_optimal_profile]
    return np.exp((total_payoff + expected_interaction_payoffs) / temp)
  return log_lik


def myopic_single_choice_likelihood(probs1, probs2, choice, payoffs, temp=1.):
  """
  Construct likelihood function for a single choice between gambles.

  :param probs1:
  :param probs2:
  :param choice:
  :param temp:
  :return:
  """
  u1 = np.sum(np.multiply(payoffs, probs1))
  u2 = np.sum(np.multiply(payoffs, probs2))
  exp_u1 = np.exp(u1 / temp)
  p1_boltzmann = exp_u1 / (exp_u1 + np.exp(u2 / temp))
  return -np.log((1-choice)*p1_boltzmann + choice*(1-p1_boltzmann))


def myopic_choice_likelihood(probs_and_choices, temp=1., penalty=1.):
  """

  :param probs_and_choices: List of tuples (probs1, probs2, choice)
  :param temp:
  :return:
  """
  def log_lik(payoffs):
    log_lik_ = 0.
    for probs1, probs2, choice in probs_and_choices:
      log_lik_ += myopic_single_choice_likelihood(probs1, probs2, choice, payoffs, temp=temp)
    return log_lik_ + penalty*np.dot(payoffs, payoffs)
  return log_lik


def maximize_likelihood(probs_and_choices, payoffs_mi_prior=np.zeros((2, 2)), temp=1.):
  log_lik = choice_likelihood(probs_and_choices, payoffs_mi_prior, temp=temp)
  x0 = np.zeros(4)
  res = minimize(log_lik, x0, method='L-BFGS-B')
  max_lik_payoffs = res.x.reshape((2,2))
  return max_lik_payoffs


def maximize_myopic_likelihood(probs_and_choices, temp=1.):
  log_lik = myopic_choice_likelihood(probs_and_choices, temp=temp)
  x0 = np.zeros(4)
  res = minimize(log_lik, x0, method='L-BFGS-B')
  max_lik_payoffs = res.x.reshape((2,2))
  return max_lik_payoffs


def max_lik_welfare(probs_and_choices_1, probs_and_choices_2, temp1=1., temp2=1.):
  max_lik_payoffs_1 = maximize_likelihood(probs_and_choices_1, temp=temp1)
  max_lik_payoffs_2 = maximize_likelihood(probs_and_choices_2, temp=temp2)
  best_act = None
  best_welfare = -float('inf')
  for a1 in range(2):
    for a2 in range(2):
      welfare = max_lik_payoffs_1[a1, a2] + max_lik_payoffs_2[a1, a2]
      if welfare > best_welfare:
        best_act = (a1, a2)
        best_welfare = welfare
  return best_act


def generate_choice_experiments(true_payoffs, n):
  probs_and_choices = []
  for rep in range(n):
    probs1 = np.random.dirichlet(np.ones(4))
    probs2 = np.random.dirichlet(np.ones(4))
    choice = np.dot(true_payoffs.flatten(), probs1) > np.dot(true_payoffs.flatten(), probs2)
    probs_and_choices.append((probs1, probs2, choice))
  return probs_and_choices


def interactions(payoffs1, payoffs2, policy1, policy2, n_interactions):
  """
  Memory-1 policies.

  :param payoffs1:
  :param payoffs2:
  :param policy1:
  :param policy2:
  :param num_interactions:
  :return:
  """
  payoffs1_list = []
  payoffs2_list = []
  s1 = 0  # Coding cooperation as 0 and defection as 1
  s2 = 0
  punish1 = False
  punish2 = False
  for rep in range(n_interactions):
    a1, m1, c1 = policy1(s1)  # Policies should return their mixed strategy (m1) and the mixed strat they deem cooperative (c1)
    a2, m2, c2 = policy2(s2)
    r1 = payoffs1[a1, a2]
    r2 = payoffs2[a1, a2]
    payoffs1_list.append(r1)
    payoffs2_list.append(r2)
    if m2 == c1 or punish2:  # Don't count defections when there's a punishment happening
      s1 = 0
    else:
      s1 = 1
      punish1 = True
    if m1 == c2 or punish1:
      s2 = 0
    else:
      s2 = 1
      punish2 = True
  total_payoffs1 = np.sum(payoffs1_list)
  total_payoffs2 = np.sum(payoffs2_list)
  pdb.set_trace()
  return total_payoffs1, total_payoffs2


def max_lik_cooperative_policy(estimated_payoffs1, estimated_payoffs2, player_ix):
  # ToDo: currently implementing pure strategies only
  best_welfare = -float('inf')
  for a1 in range(2):
    for a2 in range(2):
      welfare = estimated_payoffs1[a1, a2] + estimated_payoffs2[a1, a2]
      if welfare > best_welfare:
        best_act = (a1, a2)
        best_welfare = welfare
  if player_ix == 0:
    ai = best_act[0]
    cj = best_act[1]
  else:
    ai = best_act[1]
    cj = best_act[0]
  return ai, ai, cj


def max_lik_punishment_policy(estimated_payoffs1, estimated_payoffs2, player_ix):
  # ToDo: implement mixed strategies
  if player_ix == 0:
    estimated_payoffs_mi = estimated_payoffs2
    max_axis = 0  # ToDo: check this, brain is farting
  else:
    estimated_payoffs_mi = estimated_payoffs1
    max_axis = 1
  max_mi = np.max(estimated_payoffs_mi, axis=max_axis)
  min_max_action = np.argmax(max_mi)
  return min_max_action


def max_lik_tft_policy(s, estimated_payoffs1, estimated_payoffs2, player_ix):
  """
  
  :param s: 0 for counterpart cooperated on previous turn, 1 for defect
  :param estimated_payoffs1: 
  :param estimated_payoffs2: 
  :return: 
  """
  a_coop, aj, cj = max_lik_cooperative_policy(estimated_payoffs1, estimated_payoffs2, player_ix)
  if s:
    ai = max_lik_punishment_policy(estimated_payoffs1, estimated_payoffs2, player_ix)
  else:
    ai = a_coop
  return ai, aj, cj


def episode(payoffs1, payoffs2, interaction_policy1, interaction_policy2, n_choice_experiments, n_interactions,
            temp1, temp2):
  # Do choice experiments and estimate reward functions
  probs_and_choices1 = generate_choice_experiments(payoffs1, n_choice_experiments)
  probs_and_choices2 = generate_choice_experiments(payoffs2, n_choice_experiments)
  # estimated_payoffs_ij = player j's estimate of player i's reward function
  player_mi_prior_11 = np.random.normal(scale=0.5, size=(2, 2))
  player_mi_prior_21 = np.random.normal(scale=0.5, size=(2, 2))
  player_mi_prior_12 = np.random.normal(scale=0.5, size=(2, 2))
  player_mi_prior_22 = np.random.normal(scale=0.5, size=(2, 2))
  estimated_payoffs_11 = maximize_likelihood(probs_and_choices1, payoffs_mi_prior=player_mi_prior_11, temp=temp1)
  estimated_payoffs_21 = maximize_likelihood(probs_and_choices2, payoffs_mi_prior=player_mi_prior_21, temp=temp1)
  estimated_payoffs_12 = maximize_likelihood(probs_and_choices1, payoffs_mi_prior=player_mi_prior_12, temp=temp2)
  estimated_payoffs_22 = maximize_likelihood(probs_and_choices2, payoffs_mi_prior=player_mi_prior_22, temp=temp2)

  # Fix interaction policies and run interaction phase
  policy1 = lambda s: interaction_policy1(s, estimated_payoffs_11, estimated_payoffs_21, 0)
  policy2 = lambda s: interaction_policy2(s, estimated_payoffs_12, estimated_payoffs_22, 1)
  total_payoffs1, total_payoffs2 = interactions(payoffs1, payoffs2, policy1, policy2, n_interactions)

  # ToDo: add payoffs from choice experiments
  return total_payoffs1, total_payoffs2


if __name__ == "__main__":
  pd_payoffs1 = np.array([[-1., -3.], [0., -2.]])
  pd_payoffs2 = np.array([[-1., 0.], [-3., -2.]])
  print(episode(pd_payoffs1, pd_payoffs2, max_lik_tft_policy, max_lik_tft_policy, 5, 5, 1., 2.))

















