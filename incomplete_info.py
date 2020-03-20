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


def single_choice_likelihood(probs1, probs2, choice, payoffs, temp=1.):
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


def choice_likelihood(probs_and_choices, temp=1.):
  """

  :param probs_and_choices: List of tuples (probs1, probs2, choice)
  :param temp:
  :return:
  """
  def log_lik(payoffs):
    log_lik_ = 0.
    for probs1, probs2, choice in probs_and_choices:
      log_lik_ += single_choice_likelihood(probs1, probs2, choice, payoffs, temp=temp)
    return log_lik_
  return log_lik


def maximize_likelihood(probs_and_choices, temp=1.):
  log_lik = choice_likelihood(probs_and_choices, temp=temp)
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










