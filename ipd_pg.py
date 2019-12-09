"""
Iterated games with policy gradient learning.
"""

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad, set_detect_anomaly
from random import uniform
from functools import partial
from scipy.special import expit, logit
from abc import ABCMeta, abstractmethod
import pdb
import optim
import random
import sys

set_detect_anomaly(True)


def check_exploitability(reward_1, reward_2, player):
  # Get estimated bargaining profile
  best_sum = -float('inf')
  best_profile = None

  for i in range(2):
    for j in range(2):
      reward_sum = reward_1[i, j] + reward_2[j, i]
      if reward_sum > best_sum:
        best_profile = (i, j)
        best_sum = reward_sum

  if player == 1:
    security = np.max((np.min(reward_1[:, 0]), np.min(reward_1[:, 1])))
    return security > reward_1[best_profile[0], best_profile[1]]
  elif player == 2:
    security = np.max((np.min(reward_2[:, 0]), np.min(reward_2[:, 1])))
    return security > reward_1[best_profile[1], best_profile[0]]


def enforceability_ht(reward_1_estimates, reward_2_estimates, reward_1_counts, reward_2_counts,
                      cutoff=0.95, sampling_dbn_draws=1000):
  """
  Test the hypothesis that the utilitarian welfare function is enforceable.

  :param reward_1_estimates:
  :param reward_2_estimates:
  :param reward_1_counts:
  :param reward_2_counts:
  :return:
  """
  # Get estimated bargaining profile
  best_sum = -float('inf')
  best_profile = None

  for i in range(2):
    for j in range(2):
      reward_sum = reward_1_estimates[i, j] + reward_2_estimates[j, i]
      if reward_sum > best_sum:
        best_profile = (i, j)
        best_sum = reward_sum

  player_1_security = np.max((np.min(reward_1_estimates[:, 0]), np.min(reward_1_estimates[:, 1])))
  player_2_security = np.max((np.min(reward_2_estimates[:, 0]), np.min(reward_2_estimates[:, 1])))

  # If estimated profile is exploitable, test exploitability hypothesis
  # ToDo: assuming known variance!
  player_1_exploitable = player_2_exploitable = False # Using exploitable to mean non-enforceable
  num_non_enforceable = 0.
  if player_1_security > reward_1_estimates[best_profile[0], best_profile[1]]:
    sampling_dbn_draws_1 = np.random.normal(loc=reward_1_estimates, scale=0.1 / np.sqrt(np.maximum(reward_1_counts, 1.)),
                                            size=(sampling_dbn_draws, 2, 2))
    sampling_dbn_draws_2 = np.random.normal(loc=reward_2_estimates, scale=0.1 / np.sqrt(np.maximum(reward_2_counts, 1.)),
                                            size=(sampling_dbn_draws, 2, 2))
    for rhat_1, rhat_2 in zip(sampling_dbn_draws_1, sampling_dbn_draws_2):
      num_non_enforceable += check_exploitability(rhat_1, rhat_2, 1)
    player_2_exploitable = (num_non_enforceable / sampling_dbn_draws) > cutoff
  elif player_2_security > reward_2_estimates[best_profile[1], best_profile[0]]:
    sampling_dbn_draws_1 = np.random.normal(loc=reward_1_estimates, scale=0.1 / np.sqrt(np.maximum(reward_1_counts, 1.)),
                                            size=(sampling_dbn_draws, 2, 2))
    sampling_dbn_draws_2 = np.random.normal(loc=reward_2_estimates, scale=0.1 / np.sqrt(np.maximum(reward_2_counts, 1.)),
                                            size=(sampling_dbn_draws, 2, 2))
    for rhat_1, rhat_2 in zip(sampling_dbn_draws_1, sampling_dbn_draws_2):
      num_non_enforceable += check_exploitability(rhat_1, rhat_2, 2)
    player_1_exploitable = (num_non_enforceable / sampling_dbn_draws) > cutoff
  return player_1_exploitable, player_2_exploitable


class PD_PGLearner(metaclass=ABCMeta):
  def __init__(self,
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
    self.payoffs1_cat = T.cat((T.tensor(payoffs1[0, :]), T.tensor(payoffs2[1, :])), 0).float()
    self.payoffs2_cat = T.cat((T.tensor(payoffs2[:, 0]), T.tensor(payoffs2[:, 1])), 0).float()
    self.payoffs1_log = None
    self.payoffs2_log = None
    self.defect1 = False
    self.defect2 = False
    self.reward_history = None
    self.action_history = None
    self.state_history = None
    self.ipw_history = None
    self.opponent_reward_estimates_1 = None # For punishment policy
    self.opponent_reward_estimates_2 = None
    self.current_state = None
    self.a_punish_1 = None
    self.a_punish_2 = None
    self.opponent_reward_estimate_counts_1 = None
    self.opponent_reward_estimate_counts_2 = None

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
    p_a1 = probs1[a1]
    p_a2 = probs2[a2]
    ipw_1 = 1. / p_a1
    ipw_2 = 1. / p_a2
    ipw = ipw_1 * ipw_2

    return a1, a2, p_a1, p_a2, ipw

  def outcomes(self, a1, a2):
    # Draw rewards
    mu1 = self.payoffs1[a1, a2]
    mu2 = self.payoffs2[a2, a1]
    r1 = np.random.normal(mu1, 0.1)
    r2 = np.random.normal(mu2, 0.1)

    return r1, r2

  def bargaining_updates(self, lr, params1, params2, bargaining_updater, V, suboptimality_tolerance):
    """
    Get updates from bargaining updaters (in order to check whether these were actually followed.)

    :return:
    """
    # Get correct updates
    dVd1, dVd2 = bargaining_updater(V((params1, params2)), (params1, params2), create_graph=True)
    bargaining_update1 = (lr * dVd1).data
    bargaining_update2 = (lr * dVd2).data
    return bargaining_update1, bargaining_update2

  def gradient_ascent(self,
                      lr,
                      lr_opponent,
                      params1,
                      params2,
                      updater1,
                      updater2,
                      defect1,
                      defect2,
                      V,
                      V1,
                      V2,
                      R_opponent_1,
                      R_opponent_2):
    if lr_opponent is None:
      lr_opponent = lr

    # Get actual updates
    # Currently assuming punishment policy of the form minmax Reward_estimator
    update1, a_punish_1 = updater1(V, V1, V2, params1, params2, lr, 1, defect2, R_opponent_1)
    update2, a_punish_2 = updater2(V, V1, V2, params1, params2, lr, 2, defect1, R_opponent_2)
    return update1, update2, a_punish_1, a_punish_2

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
                      observable_seed=True,
                      cutoff=0.1,
                      exploitability_policy=None, # Must be supplied if hypothesis_test=True
                      **kwargs,  # these are forwarded to the parameters-to-outcomes function
                      ):
    pr_CC = []
    pr_DD = []
    payoffs1 = []
    payoffs2 = []
    step_list = []
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
                          observable_seed=observable_seed,
                          cutoff=cutoff,
                          exploitability_policy=exploitability_policy, # Must be supplied if hypothesis_test=True
                          **kwargs  # these are forwarded to the parameters-to-outcomes function
                          )
      if plot_learning:
        pr_CC.append(results['pr_CC'])
        pr_DD.append(results['pr_DD'])
        step_list.append(np.arange(n_epochs))
        payoffs1.append(results['payoffs1'])
        payoffs2.append(results['payoffs2'])

    if plot_learning:
      # Get average values over each replicate
      self.pr_CC_log = np.hstack(pr_CC)
      self.pr_DD_log = np.hstack(pr_DD)
      self.payoffs1_log = np.array(payoffs1).mean(axis=0)
      self.payoffs2_log = np.array(payoffs2).mean(axis=0)
      self.step_list = np.hstack(step_list)
      self.payoffs1_log = np.hstack(payoffs1)
      self.payoffs2_log = np.hstack(payoffs2)

      # Plot
      self.plot_last_learning(label, multi_rep=True)

  def maximum_likelihood(self, no_punish_ixs_1_, no_punish_ixs_2_):
    # look_back = np.min((look_back, len(self.action_history)-1))
    # Get maximum likelihood stationary policies
    # Player 1
    cooperate_in_state_c_1 = np.array([ah[0] for ix, ah in enumerate(self.action_history[:-1])
                                       if self.state_history[ix+1][0] == 0 and ix in no_punish_ixs_1_])
    cooperate_in_state_d_1 = np.array([ah[0] for ix, ah in enumerate(self.action_history[:-1])
                                       if self.state_history[ix+1][0] == 1 and ix in no_punish_ixs_1_])

    # Player 2
    cooperate_in_state_c_2 = np.array([ah[1] for ix, ah in enumerate(self.action_history[:-1])
                                       if self.state_history[ix+1][1] == 0 and ix in no_punish_ixs_2_])
    cooperate_in_state_d_2 = np.array([ah[1] for ix, ah in enumerate(self.action_history[:-1])
                                       if self.state_history[ix+1][1] == 1 and ix in no_punish_ixs_2_])

    # Max likelihood parameters
    p1_c, p1_d = np.mean(1-cooperate_in_state_c_1), np.mean(1-cooperate_in_state_d_1)
    p2_c, p2_d = np.mean(1-cooperate_in_state_c_2), np.mean(1-cooperate_in_state_d_2)

    # Likelihoods
    n_cc_1 = len(cooperate_in_state_c_1)
    f_cc_1 = np.sum(cooperate_in_state_c_1)
    n_dc_1 = len(cooperate_in_state_d_1)
    f_dc_1 = np.sum(cooperate_in_state_d_1)
    n_cc_2 = len(cooperate_in_state_c_2)
    f_cc_2 = np.sum(cooperate_in_state_c_2)
    n_dc_2 = len(cooperate_in_state_d_2)
    f_dc_2 = np.sum(cooperate_in_state_d_2)

    max_lik_1 = np.power(p1_c, n_cc_1 - f_cc_1) * np.power(1-p1_c, f_cc_1) * np.power(p1_d, n_dc_1 - f_dc_1) * np.power(1-p1_d, f_dc_1)
    max_lik_2 = np.power(p2_c, n_cc_2 - f_cc_2) * np.power(1-p2_c, f_cc_2) * np.power(p2_d, n_dc_2 - f_dc_2) * np.power(1-p2_d, f_dc_2)
    return max_lik_1, max_lik_2

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
            hypothesis_test=False,
            exploitability_policy=None, # Must be supplied if hypothesis_test=True,
            observable_seed=True,
            cutoff=0.1,
            **kwargs,  # these are forwarded to the parameters-to-outcomes function
            ):
    self.pr_CC_log = np.empty(n_epochs)
    self.pr_DD_log = np.empty(n_epochs)
    self.payoffs1_log = np.empty(n_epochs)
    self.payoffs2_log = np.empty(n_epochs)
    self.observed_action_probs1 = []
    self.observed_action_probs2 = []
    self.bargaining_action_probs1 = []
    self.bargaining_action_probs2 = []
    self.reward_history = []
    self.action_history = []
    self.cooperative_likelihood_history = []
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

    bargaining_params1 = std * T.randn(self.num_params1)
    if init_params1:
      assert len(init_params1) == self.num_params1, \
        "initial parameters for player 1 don't have correct length"
      bargaining_params1 += T.tensor(init_params1).float()
    bargaining_params1.requires_grad_()

    bargaining_params2 = std * T.randn(self.num_params2)
    if init_params2:
      assert len(init_params2) == self.num_params2, \
        "initial parameters for player 2 don't have correct length"
      bargaining_params2 += T.tensor(init_params2).float()
    bargaining_params2.requires_grad_()

    defect_lik_1 = 1.
    defect_lik_2 = 1.
    bargaining_probs1 = bargaining_probs2 = [0.5, 0.5]
    no_punish_ixs_1 = [0]
    no_punish_ixs_2 = [0]
    for i in range(n_epochs):
      print(i)

      # Get actions from current policy
      if not (player_1_exploitable_ or player_2_exploitable_):  # If neither is exploitable, follow param policies
        a1, a2, p_a1, p_a2, ipw = self.actions_from_params(params1, params2, self.current_state[0], self.current_state[1])
        self.observed_action_probs1.append(p_a1)
        self.observed_action_probs2.append(p_a2)
      else:
        a1, a2, ipw = exploitability_policy(self.opponent_reward_estimates_2, self.opponent_reward_estimates_1,
                                            player_1_exploitable_, player_2_exploitable_)

      # Override actions with exploration or punishment
      # if np.random.random() < 0.05:
      #  a1 = np.random.choice(2)
      # if np.random.random() < 0.05:
      #   a2 = np.random.choice(2)
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

      # Conduct hypothesis test
      if hypothesis_test:
        player_1_exploitable_,player_2_exploitable_ = \
          enforceability_ht(self.opponent_reward_estimates_2, self.opponent_reward_estimates_1,
          self.opponent_reward_estimate_counts_2, self.opponent_reward_estimate_counts_1, cutoff=0.8,
                            sampling_dbn_draws=1000)
      print(self.opponent_reward_estimates_2, self.opponent_reward_estimates_1)
      print(player_1_exploitable_, player_2_exploitable_)

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
        V1_, V2_ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
        return (V1_ + V2_).requires_grad_()

      def V1(p1p2):
        p1, p2 = p1p2
        V1_, _ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
        return V1_.requires_grad_()

      def V2(p1p2):
        p1, p2 = p1p2
        _, V2_ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
        return V2_.requires_grad_()

      print(a1)
      self.payoffs1_log[i] = self.payoffs1[a1, a2]
      self.payoffs2_log[i] = self.payoffs2[a2, a1]

      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      # noinspection PyInterpreter

      # Define opponent reward functions for punishment policy
      def R_opponent_1(a2_, a1_):
        return self.opponent_reward_estimates_1[a2_, a1_]

      def R_opponent_2(a1_, a2_):
        return self.opponent_reward_estimates_2[a1_, a2_]

      # Get each agent's update
      update1, update2, self.a_punish_1, self.a_punish_2 = self.gradient_ascent(
        lr,
        lr_opponent,
        params1,
        params2,
        updater1,
        updater2,
        self.defect1,
        self.defect2,
        V,
        V1,
        V2,
        R_opponent_1,
        R_opponent_2
      )

      # Compare to bargaining updates
      bargaining_update1, bargaining_update2 = self.bargaining_updates(lr, params1, params2, bargaining_updater, V,
                                                                         suboptimality_tolerance)
      if observable_seed:
        V_bargaining = V((params1 + bargaining_update1, params2 + bargaining_update2)).detach().numpy()

        if np.abs(V_bargaining - V((params1 + bargaining_update1, params2 + update2)).detach().numpy()) \
                < suboptimality_tolerance:
          self.defect2 = False
        else:
          self.defect2 = True
        if np.abs(V_bargaining - V((params1 + update1, params2 + bargaining_update2)).detach().numpy()) \
                < suboptimality_tolerance:
          self.defect1 = False
        else:
          self.defect1 = True
        print(self.defect1, self.defect2)
      else: # Infer whether you are being defected against

        p_a1_barg = bargaining_probs1[a1]
        p_a2_barg = bargaining_probs2[a2]
        self.bargaining_action_probs1.append(p_a1_barg)
        self.bargaining_action_probs2.append(p_a2_barg)
        self.cooperative_likelihood_history.append([p_a1_barg, p_a2_barg])
        LOOK_BACK = 50
        look_back = np.min((LOOK_BACK, i+1))
        likelihood_coop_1 = np.prod(np.array(self.cooperative_likelihood_history)[no_punish_ixs_2[-look_back:-1], 0])
        likelihood_coop_2 = np.prod(np.array(self.cooperative_likelihood_history)[no_punish_ixs_1[-look_back:-1], 1])
        max_lik_stationary_1, max_lik_stationary_2 = self.maximum_likelihood(no_punish_ixs_2[-look_back:], no_punish_ixs_1[-look_back:])
        TOL = 0.5
        if i > 50:
          # ratios_1 = []
          # ratios_2 = []
          # for b in range(100):
          #   no_punish_ixs_1_b = np.random.choice(no_punish_ixs_1[-look_back], look_back)
          #   no_punish_ixs_2_b = np.random.choice(no_punish_ixs_2[-look_back], look_back)
          #   likelihood_coop_1_b = np.prod(np.array(self.cooperative_likelihood_history)[no_punish_ixs_2_b[:-1], 0])
          #   likelihood_coop_2_b = np.prod(np.array(self.cooperative_likelihood_history)[no_punish_ixs_1_b[:-1], 1])
            # ToDo: also need to pass indices of bootstrapped... indices to max_lik
          #   max_lik_stationary_1_b, max_lik_stationary_2_b = self.maximum_likelihood(no_punish_ixs_2_b,
          #                                                                           no_punish_ixs_1_b)
          #   ratios_1.append(max_lik_stationary_1_b / likelihood_coop_1)
          #   ratios_2.append(max_lik_stationary_2_b / likelihood_coop_2)
          if i == 51:
            initial_defect_lik_1 = max_lik_stationary_1 / likelihood_coop_1
            initial_defect_lik_2 = max_lik_stationary_2 / likelihood_coop_2
          if np.log(max_lik_stationary_1)/len(no_punish_ixs_2) - np.log(likelihood_coop_1)/len(no_punish_ixs_2) >= 1 / (i - 50):
            self.defect1 = True
            defect_lik_1 = max_lik_stationary_1 /likelihood_coop_1
          else:
            self.defect1 = False
            no_punish_ixs_1.append(i+1)
          if np.log(max_lik_stationary_2)/len(no_punish_ixs_1) - np.log(likelihood_coop_2)/len(no_punish_ixs_1) >= 1 / (i - 50):
            self.defect2 = True
            defect_lik_2 = max_lik_stationary_2 / likelihood_coop_2
          else:
            self.defect2 = False
            no_punish_ixs_2.append(i+1)
        else:
          no_punish_ixs_1.append(i+1)
          no_punish_ixs_2.append(i+1)
        bargaining_probs1 = T.sigmoid((bargaining_params1 + bargaining_update1)[(2 * a2):(2 * a2 + 2)]).detach().numpy()
        bargaining_probs2 = T.sigmoid((bargaining_params2 + bargaining_update2)[(2 * a1):(2 * a1 + 2)]).detach().numpy()
        bargaining_probs1 /= np.sum(bargaining_probs1)
        bargaining_probs2 /= np.sum(bargaining_probs2)

      # Do updates
      params1.data += update1
      params2.data += update2
      bargaining_params1.data += bargaining_update1
      bargaining_params2.data += bargaining_update2

    self.final_params = (params1, params2)
    if plot_learning:
      self.plot_last_learning()
    return {'pr_CC':self.pr_CC_log, 'pr_DD': self.pr_DD_log, 'payoffs1': self.payoffs1_log,
            'payoffs2': self.payoffs2_log}

  def plot_last_learning(self, label, multi_rep):
    if self.pr_CC_log is not None:
      if multi_rep:
        # Plot prob time series
        fig, axs = plt.subplots(nrows=3)
        payoffs = np.hstack((self.payoffs1_log, self.payoffs2_log))
        probs_series_df = {'prob': np.hstack((self.pr_CC_log, self.pr_DD_log)),
                           'steps': np.hstack((self.step_list, self.step_list)),
                           'profile': np.hstack((['CC'] * len(self.step_list), ['DD'] * len(self.step_list))),
                           'payoffs': payoffs,
                           'cum_payoffs': np.hstack((np.cumsum(self.payoffs1_log) / np.arange(1, len(self.payoffs1_log)+1),
                                                     np.cumsum(self.payoffs2_log) / np.arange(1, len(self.payoffs2_log)+1))),

                           'player': np.hstack(
                             (['player 1'] * len(self.step_list), ['player 2'] * len(self.step_list)))}
        probs_series_df = pd.DataFrame(probs_series_df)
        sns.lineplot(x='steps', y='prob', hue='profile', data=probs_series_df, ax=axs[0])

        # Plot payoffs time series
        sns.lineplot(x='steps', y='payoffs', hue='player', data=probs_series_df, ax=axs[1])

        # Cumulative payoffs time series
        sns.lineplot(x='steps', y='cum_payoffs', hue='player', data=probs_series_df, ax=axs[2])

        if label is not None: # save figure if filename given
          plt.savefig('{}.png'.format(label))
        else:
          plt.show()
      else:
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


class IPD_PG(PD_PGLearner):
  """
  IPD with policy gradient instead of exact solution.
  """

  def __init__(self, payoffs1, payoffs2):
    super().__init__(payoffs1=payoffs1, payoffs2=payoffs2)
    self.num_params1 = 4
    self.num_params2 = 4

  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history,
              exact=False, gamma=0.99):
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

    if exact:
      P = T.stack([
        probs1 * probs2,
        probs1 * (1 - probs2),
        (1 - probs1) * probs2,
        (1 - probs1) * (1 - probs2)
      ])
      s0 = T.tensor([1., 0., 0., 0.])
      transition_matrix = P
      infi_sum = T.inverse(T.eye(4) - gamma * transition_matrix)
      avg_state = (1 - gamma) * T.matmul(infi_sum, s0)
      return T.dot(avg_state, self.payoffs1_cat), T.dot(avg_state, self.payoffs2_cat)
    else:
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


if __name__ == "__main__":
  pd_payoffs1 = np.array([[-1., -3.], [0., -2.]])
  pd_payoffs2 = np.array([[-1., -3.], [0., -2.]])
  no_enforce_payoffs_1 = np.array([[0., -1.], [-1., -0.75]])
  no_enforce_payoffs_2 = np.array([[2., 2.], [2.5, 2.5]])
  stag_payoffs1 = np.array([[2., -3.], [0., 1.]])
  stag_payoffs2 = np.array([[2., -3.], [0., 1.]])

  ipd = IPD_PG(payoffs1=pd_payoffs1, payoffs2=pd_payoffs2)
  ipd.learn_multi_rep('pd-private-tft-2', 20, 1.0, optim.gradient_ascent_minmax_reward,
                    optim.gradient_ascent_minmax_reward, grad, observable_seed=False, n_epochs=1000)
  ipd.learn_multi_rep('pd-private-tft-naive-2', 20, 1.0, optim.gradient_ascent_minmax_reward,
                      optim.naive_gradient_ascent, grad, observable_seed=False, n_epochs=1000)

  # no_enforce = IPD_PG(payoffs1=no_enforce_payoffs_1, payoffs2=no_enforce_payoffs_2)
  # no_enforce.learn_multi_rep('game-2-with-ht', 20, 0.5, optim.gradient_ascent_minmax_reward,
  #                            optim.gradient_ascent_minmax_reward, grad,
  #                            n_epochs=5000, hypothesis_test=True,
  #                            exploitability_policy=optim.max_min_exploitability_policy)
  # no_enforce.learn_multi_rep('game-2-without-ht', 20, 0.5, optim.gradient_ascent_minmax_reward,
  #                            optim.gradient_ascent_minmax_reward, grad, n_epochs=5000)
