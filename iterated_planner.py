"""
Simulation optimization for finding bargaining policies (rather than bargaining updates).
"""
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad, set_detect_anomaly
from random import uniform
from functools import partial
from scipy.special import expit
from abc import ABCMeta, abstractmethod
import pdb
import optim
import random
import sys

set_detect_anomaly(True)


def defection_test(player_1_reward_history_, player_2_reward_history_, player_1_cooperative_reward_mean,
                   player_2_cooperative_reward_mean, cutoff):
  """

  :param player_1_reward_history_: Player 1's observed rewards
  :param player_2_reward_history_: Player 2's observed rewards
  :param player_1_cooperative_reward_mean: Expected value under cooperation for player 1
  :param player_2_cooperative_reward_mean: Expected value under cooperation for player 1
  :return:
  """
  player_2_defecting = (np.mean(player_1_reward_history_) - player_1_cooperative_reward_mean) / \
                       np.std(player_1_reward_history_) < cutoff
  player_1_defecting = (np.mean(player_2_reward_history_) - player_2_cooperative_reward_mean) / \
                       np.std(player_2_reward_history_) < cutoff
  return player_1_defecting, player_2_defecting


# ToDo: implement Solver parent class to eliminate redundancies in the classes below
class MaxMinSolver(metaclass=ABCMeta):
  def __init__(self,
               payoffs1=np.array([[-1., -3.], [0., -2.]]),
               **kwargs):
    # overwrite these in child classes; **kwargs can be used here
    self.num_params1 = -1
    self.num_params2 = -1
    # generate these when learning
    self.pr_CC_log = None
    self.pr_DD_log = None
    self.payoffs1 = payoffs1
    self.payoffs2 = -payoffs1
    self.payoffs1_log = None
    self.payoffs2_log = None
    self.reward_history = None
    self.action_history = None
    self.state_history = None
    self.ipw_history = None

  def actions_from_params(self, params1, params2, s1, s2):
    # Draw actions from policies
    probs1 = T.sigmoid(params1[(2*s1):(2*s1+2)]).detach().numpy()
    probs2 = T.sigmoid(params2[(2*s2):(2*s2+2)]).detach().numpy()
    probs1 /= np.sum(probs1)
    probs2 /= np.sum(probs2)
    a1 = np.random.choice(range(2), p=probs1)
    a2 = np.random.choice(range(2), p=probs2)

    # Get ipws
    ipw_1 = 1. / probs1[a1]
    ipw_2 = 1. / probs2[a2]
    ipw = ipw_1 * ipw_2

    return a1, a2, ipw

  def outcomes(self, a1, a2):
    # Draw rewards
    mu1 = self.payoffs1[a1, a2]
    r1 = np.random.normal(mu1, 0.1)
    r2 = -r1

    return r1, r2

  def gradient_ascent(self,
                      lr,
                      lr_opponent,
                      params1,
                      params2,
                      updater1,
                      updater2,
                      V):
    if lr_opponent is None:
      lr_opponent = lr

    # Get actual updates
    update1 = updater1(V, params1, params2, lr, 1)
    update2 = updater2(V, params1, params2, lr, 2)

    return update1, update2

  def learn(self,
            lr,
            updater1,
            updater2,
            lr_opponent=None,
            std=0.01,
            n_epochs=2000,
            n_print_every=None,
            init_params1=None,
            init_params2=None,
            plot_learning=True,
            **kwargs,  # these are forwarded to the parameters-to-outcomes function
            ):
    self.pr_CC_log = np.empty(n_epochs)
    self.pr_DD_log = np.empty(n_epochs)
    self.payoffs1_log = np.empty(n_epochs)
    self.payoffs2_log = np.empty(n_epochs)
    self.reward_history = []
    self.action_history = []
    self.state_history = []
    self.ipw_history = []
    self.opponent_reward_estimates_1 = np.zeros((2, 2))  # Agent 2 reward at (a_2, a_1)
    self.opponent_reward_estimates_2 = np.zeros((2, 2))  # Agent 1 reward at (a_1, a_2)
    self.opponent_reward_estimate_counts_1 = np.zeros((2, 2))
    self.opponent_reward_estimate_counts_2 = np.zeros((2, 2))
    self.current_state = (0, 0) # Start off both cooperating

    params1 = std * T.randn(self.num_params1)
    if init_params1:
      assert len(init_params1) == self.num_params1, \
        "initial parameters for player 1 don't have correct length"
      params1 += T.tensor(init_params1).float()
    params1.requires_grad_()

    params2 = std * T.randn(self.num_params2)
    if init_params2:
      assert len(init_params2) == self.num_params2, \
        "initial parameters for player 2 don't have correct length"
      params2 += T.tensor(init_params2).float()
    params2.requires_grad_()

    for i in range(n_epochs):
      print(i)

      a1, a2, ipw = self.actions_from_params(params1, params2, self.current_state[0], self.current_state[1])

      # Observe rewards
      r1, r2 = self.outcomes(a1, a2)

      # Update histories
      self.state_history.append(self.current_state)
      self.reward_history.append((r1, r2))
      self.action_history.append((a1, a2))
      self.ipw_history.append(ipw)
      self.opponent_reward_estimate_counts_1[a2, a1] += 1
      self.opponent_reward_estimate_counts_2[a1, a2] += 1
      self.opponent_reward_estimates_1[a2, a1] += (r2 - self.opponent_reward_estimates_1[a2, a1]) / \
        self.opponent_reward_estimate_counts_1[a2, a1]
      self.opponent_reward_estimates_2[a1, a2] += (r1 - self.opponent_reward_estimates_2[a1, a2]) / \
        self.opponent_reward_estimate_counts_2[a1, a2]
      self.current_state = (a2, a1) # Agent i's state is agent -i's previous action

      probs1 = T.sigmoid(params1[(2*a2):(2*a2 + 2)]).detach().numpy()
      probs2 = T.sigmoid(params2[(2*a1):(2*a1 + 2)]).detach().numpy()
      probs1 /= np.sum(probs1)
      probs2 /= np.sum(probs2)
      self.pr_CC_log[i] = probs1[0]*probs2[0]
      self.pr_DD_log[i] = probs1[1]*probs2[1]

      # Define bargaining value function estimator
      # V1, V2 = self.payoffs(params1, params2, ipw_history, reward_history, action_history, state_history)
      def V(p1p2):
        p1, p2 = p1p2
        V_ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
        return V_.requires_grad_()

      self.payoffs1_log[i] = self.payoffs1[a1, a2]
      self.payoffs2_log[i] = self.payoffs2[a2, a1]

      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      # noinspection PyInterpreter

      # Get each agent's update
      update1, update2 = self.gradient_ascent(
        lr,
        lr_opponent,
        params1,
        params2,
        updater1,
        updater2,
        V,
      )

      # Do updates
      params1.data += update1
      params2.data += update2

    self.final_params = (params1, params2)
    if plot_learning:
      self.plot_last_learning()
    return {'pr_CC':self.pr_CC_log, 'pr_DD': self.pr_DD_log, 'payoffs1': self.payoffs1_log,
            'payoffs2': self.payoffs2_log}


class NashBargainingSolver(metaclass=ABCMeta):
  def __init__(self,
               disagreement_value_1,
               disagreement_value_2,
               payoffs1=np.array([[-1., -3.], [0., -2.]]),
               payoffs2=np.array([[-1., -3.], [0., -2.]]),
               **kwargs):
    # overwrite these in child classes; **kwargs can be used here
    self.num_params1 = -1
    self.num_params2 = -1
    # generate these when learning
    self.pr_CC_log = None
    self.pr_DD_log = None
    self.payoffs1 = payoffs1
    self.payoffs2 = payoffs2
    self.payoffs1_log = None
    self.payoffs2_log = None
    self.reward_history = None
    self.action_history = None
    self.state_history = None
    self.ipw_history = None
    self.disagreement_value_1 = disagreement_value_1
    self.disagreement_value_2 = disagreement_value_2

  @abstractmethod
  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    pass

  def actions_from_params(self, params1, params2, s1, s2):
    # Draw actions from policies
    probs1 = T.sigmoid(params1[(2*s1):(2*s1+2)]).detach().numpy()
    probs2 = T.sigmoid(params2[(2*s2):(2*s2+2)]).detach().numpy()
    probs1 /= np.sum(probs1)
    probs2 /= np.sum(probs2)
    a1 = np.random.choice(range(2), p=probs1)
    a2 = np.random.choice(range(2), p=probs2)

    # Get ipws
    ipw_1 = 1. / probs1[a1]
    ipw_2 = 1. / probs2[a2]
    ipw = ipw_1 * ipw_2

    return a1, a2, ipw

  def outcomes(self, a1, a2):
    # Draw rewards
    mu1 = self.payoffs1[a1, a2]
    mu2 = self.payoffs2[a2, a1]
    r1 = np.random.normal(mu1, 0.1)
    r2 = np.random.normal(mu2, 0.1)

    return r1, r2

  def gradient_ascent(self,
                      lr,
                      lr_opponent,
                      params1,
                      params2,
                      updater1,
                      updater2,
                      V):
    if lr_opponent is None:
      lr_opponent = lr

    # Get actual updates
    update1 = updater1(V, params1, params2, lr, 1)
    update2 = updater2(V, params1, params2, lr, 2)

    return update1, update2

  def learn_multi_rep(self,
                      label,
                      n_rep,
                      lr,
                      updater1,
                      updater2,
                      bargaining_updater,
                      lr_opponent=None,
                      std=0.01,
                      n_epochs=2000,
                      n_print_every=None,
                      init_params1=None,
                      init_params2=None,
                      plot_learning=True,
                      suboptimality_tolerance=0.1,
                      hypothesis_test=False,
                      exploitability_policy=None, # Must be supplied if hypothesis_test=True
                      **kwargs,  # these are forwarded to the parameters-to-outcomes function
                      ):
    pr_CC = []
    pr_DD = []
    payoffs1 = []
    payoffs2 = []
    for rep in range(n_rep):
      results = self.learn(lr,
                          updater1,
                          updater2,
                          bargaining_updater,
                          lr_opponent=lr_opponent,
                          std=std,
                          n_epochs=n_epochs,
                          n_print_every=n_print_every,
                          init_params1=init_params1,
                          init_params2=init_params2,
                          plot_learning=False,
                          suboptimality_tolerance=suboptimality_tolerance,
                          hypothesis_test=hypothesis_test,
                          exploitability_policy=exploitability_policy, # Must be supplied if hypothesis_test=True
                          **kwargs  # these are forwarded to the parameters-to-outcomes function
                          )
      if plot_learning:
        pr_CC.append(results['pr_CC'])
        pr_DD.append(results['pr_DD'])
        payoffs1.append(results['payoffs1'])
        payoffs2.append(results['payoffs2'])

    if plot_learning:
      # Get average values over each replicate
      self.pr_CC_log = np.array(pr_CC).mean(axis=0)
      self.pr_DD_log = np.array(pr_DD).mean(axis=0)
      self.payoffs1_log = np.array(payoffs1).mean(axis=0)
      self.payoffs2_log = np.array(payoffs2).mean(axis=0)

      # Plot
      self.plot_last_learning(label)

  def learn(self,
            lr,
            updater1,
            updater2,
            bargaining_updater,
            lr_opponent=None,
            std=0.01,
            n_epochs=2000,
            n_print_every=None,
            init_params1=None,
            init_params2=None,
            plot_learning=True,
            suboptimality_tolerance=0.1,
            test_for_defection=True,
            exploitability_policy=None, # Must be supplied if hypothesis_test=True
            **kwargs,  # these are forwarded to the parameters-to-outcomes function
            ):
    self.pr_CC_log = np.empty(n_epochs)
    self.pr_DD_log = np.empty(n_epochs)
    self.payoffs1_log = np.empty(n_epochs)
    self.payoffs2_log = np.empty(n_epochs)
    self.reward_history = []
    self.action_history = []
    self.state_history = []
    self.ipw_history = []
    self.opponent_reward_estimates_1 = np.zeros((2, 2))  # Agent 2 reward at (a_2, a_1)
    self.opponent_reward_estimates_2 = np.zeros((2, 2))  # Agent 1 reward at (a_1, a_2)
    self.opponent_reward_estimate_counts_1 = np.zeros((2, 2))
    self.opponent_reward_estimate_counts_2 = np.zeros((2, 2))
    self.current_state = (0, 0) # Start off both cooperating
    player_1_exploitable_ = player_2_exploitable_ = False

    params1 = std * T.randn(self.num_params1)
    if init_params1:
      assert len(init_params1) == self.num_params1, \
        "initial parameters for player 1 don't have correct length"
      params1 += T.tensor(init_params1).float()
    params1.requires_grad_()

    params2 = std * T.randn(self.num_params2)
    if init_params2:
      assert len(init_params2) == self.num_params2, \
        "initial parameters for player 2 don't have correct length"
      params2 += T.tensor(init_params2).float()
    params2.requires_grad_()

    for i in range(n_epochs):
      print(i)

      # Get actions from current policy
      if not (player_1_exploitable_ or player_2_exploitable_):  # If neither is exploitable, follow param policies
        a1, a2, ipw = self.actions_from_params(params1, params2, self.current_state[0], self.current_state[1])
      else:
        a1, a2, ipw = exploitability_policy(self.opponent_reward_estimates_2, self.opponent_reward_estimates_1,
                                            player_1_exploitable_, player_2_exploitable_)

      # Override actions with exploration or punishment
      if np.random.random() < 0.05:
        a1 = np.random.choice(2)
      if np.random.random() < 0.05:
        a2 = np.random.choice(2)
      if self.a_punish_1 is not None: # If decided to punish on the last turn, replace with punishment action
        a1 = self.a_punish_1
      if self.a_punish_2 is not None:
        a2 = self.a_punish_2

      # Observe rewards
      r1, r2 = self.outcomes(a1, a2)

      # Update histories
      self.state_history.append(self.current_state)
      self.reward_history.append((r1, r2))
      self.action_history.append((a1, a2))
      self.ipw_history.append(ipw)
      self.opponent_reward_estimate_counts_1[a2, a1] += 1
      self.opponent_reward_estimate_counts_2[a1, a2] += 1
      self.opponent_reward_estimates_1[a2, a1] += (r2 - self.opponent_reward_estimates_1[a2, a1]) / \
        self.opponent_reward_estimate_counts_1[a2, a1]
      self.opponent_reward_estimates_2[a1, a2] += (r1 - self.opponent_reward_estimates_2[a1, a2]) / \
        self.opponent_reward_estimate_counts_2[a1, a2]
      self.current_state = (a2, a1) # Agent i's state is agent -i's previous action

      # Check for defections
      if test_for_defection:
        player_1_defecting_, player_2_defecting = \
          defection_test(self.reward_history_1, self.self.reward_history_2,
          self.cooperative_reward_mean_1, self.cooperative_reward_mean_2)

      print(self.opponent_reward_estimates_2, self.opponent_reward_estimates_1)
      print(player_1_defecting_, player_2_defecting_)

      probs1 = T.sigmoid(params1[(2*a2):(2*a2 + 2)]).detach().numpy()
      probs2 = T.sigmoid(params2[(2*a1):(2*a1 + 2)]).detach().numpy()
      probs1 /= np.sum(probs1)
      probs2 /= np.sum(probs2)
      self.pr_CC_log[i] = probs1[0]*probs2[0]
      self.pr_DD_log[i] = probs1[1]*probs2[1]

      # Define bargaining value function estimator
      # V1, V2 = self.payoffs(params1, params2, ipw_history, reward_history, action_history, state_history)
      def V(p1p2):
        p1, p2 = p1p2
        V_ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
        return V_.requires_grad_()

      self.payoffs1_log[i] = self.payoffs1[a1, a2]
      self.payoffs2_log[i] = self.payoffs2[a2, a1]

      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      # noinspection PyInterpreter

      # Get each agent's update
      update1, update2 = self.gradient_ascent(
        lr,
        lr_opponent,
        params1,
        params2,
        updater1,
        updater2,
        V,
      )

      # Do updates
      params1.data += update1
      params2.data += update2

    self.final_params = (params1, params2)
    if plot_learning:
      self.plot_last_learning()
    return {'pr_CC':self.pr_CC_log, 'pr_DD': self.pr_DD_log, 'payoffs1': self.payoffs1_log,
            'payoffs2': self.payoffs2_log}

  def plot_last_learning(self, label):
    if self.pr_CC_log is not None:
      steps = np.arange(len(self.pr_CC_log))
      fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
      ax = axs[0]
      ax.plot(steps, self.pr_CC_log, label='Prob[CC]')
      ax.plot(steps, self.pr_DD_log, label='Prob[DD]')
      ax.legend()
      ax = axs[1]
      ax.plot(steps, self.payoffs1_log, label='payoffs player 1')
      ax.plot(steps, self.payoffs2_log, label='payoffs player 2')
      ax.legend()
      if label is not None: # save figure if filename given
        plt.savefig('{}.png'.format(label))
      else:
        plt.show()
    else:
      print("This learner has not learnt yet.")

  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    """

    :param ipw1:
    :param ipw2:
    :param reward_history:
    :param action_history:
    :param state_history:
    :return:
    """
    if type(params1) is np.ndarray or type(params2) is np.ndarray:
      params1 = T.from_numpy(params1).float()
      params2 = T.from_numpy(params2).float()

    probs1 = T.sigmoid(params1)
    probs2 = T.sigmoid(params2)

    value_estimate_1 = 0.
    value_estimate_2 = 0.
    is_normalizer = 0. # For stability
    look_back = np.max((0, len(ipw_history) - 10))
    for ipw, r, a, s in zip(ipw_history[look_back:], reward_history[look_back:], action_history[look_back:],
                            state_history[look_back:]):
      # Get prob of a under probs1, probs2
      # ToDo: assuming params are in order [CC, CD, DC, DD]; check this!
      s1, s2 = s
      a1, a2 = a
      prob_a1 = probs1[(s1 + a1)*(1-s1) + (s1 + a1 + 1)*s1]
      prob_a2 = probs2[(s2 + a2)*(1-s2) + (s2 + a2 + 1)*s2]
      prob_a = prob_a1 * prob_a2

      # Update value estimate
      is_weight = prob_a / ipw
      is_normalizer += is_weight
      value_estimate_1 += is_weight * r[0]
      value_estimate_2 += is_weight * r[1]

    return value_estimate_1 / is_normalizer, value_estimate_2 / is_normalizer


class IteratedNashBargaining(NashBargainingSolver):
  """
  NBS for iterated game with policy gradient instead of exact solution.
  """

  def __init__(self, payoffs1, payoffs2):
    super().__init__(payoffs1=payoffs1, payoffs2=payoffs2)
    self.num_params1 = 4
    self.num_params2 = 4

  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    """

    :param ipw1:
    :param ipw2:
    :param reward_history:
    :param action_history:
    :param state_history:
    :return:
    """
    if type(params1) is np.ndarray or type(params2) is np.ndarray:
      params1 = T.from_numpy(params1).float()
      params2 = T.from_numpy(params2).float()

    probs1 = T.sigmoid(params1)
    probs2 = T.sigmoid(params2)

    value_estimate_1 = 0.
    value_estimate_2 = 0.
    is_normalizer = 0. # For stability
    look_back = np.max((0, len(ipw_history) - 10))
    for ipw, r, a, s in zip(ipw_history[look_back:], reward_history[look_back:], action_history[look_back:],
                            state_history[look_back:]):
      # Get prob of a under probs1, probs2
      # ToDo: assuming params are in order [CC, CD, DC, DD]; check this!
      s1, s2 = s
      a1, a2 = a
      prob_a1 = probs1[(s1 + a1)*(1-s1) + (s1 + a1 + 1)*s1]
      prob_a2 = probs2[(s2 + a2)*(1-s2) + (s2 + a2 + 1)*s2]
      prob_a = prob_a1 * prob_a2

      # Update value estimate
      is_weight = prob_a / ipw
      is_normalizer += is_weight
      value_estimate_1 += r[0] * is_weight
      value_estimate_2 += r[1] * is_weight

    return np.log(value_estimate_1 / is_normalizer - self.disagreement_value_1) + \
           np.log(value_estimate_2 / is_normalizer - self.disagreement_value_2)
