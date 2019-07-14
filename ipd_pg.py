"""
Iterated games with policy gradient learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad
from random import uniform
from functools import partial
from scipy.special import expit
from abc import ABCMeta, abstractmethod
import pdb
import optim
import random
import sys



class PD_PGLearner(metaclass=ABCMeta):
  # ToDo: check that payoffs are in correct order
  def __init__(self, payoffs1=[[-1., -3.], [0., -2.]], payoffs2=[[-1., 0.], [-3., -2.]], **kwargs):
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

  @abstractmethod
  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    pass

  def outcomes(self, params1, params2, s1, s2):
    # Draw actions from policies
    probs1 = T.sigmoid(params1[(2*s1):(2*s1+1)])
    probs2 = T.sigmoid(params2[(2*s2):(2*s2+1)])
    a1 = np.random.choice(p=probs1)
    a2 = np.random.choice(p=probs2)

    # Get ipws
    ipw_1 = 1. / probs1[2*s1+a1]
    ipw_2 = 1. / probs2[2*s2+a2]

    # Draw rewards
    mu1 = self.payoffs1[a1, a2]
    mu2 = self.payoffs2[a2, a1]
    r1 = np.random.normal(mu1, 1.)
    r2 = np.random.normal(m2, 1.)

    return r1, r2, a1, a2, ipw_1, ipw_2


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
                      R_opponent_1,
                      R_opponent_2):
    if lr_opponent is None:
      lr_opponent = lr

    # Get actual updates
    # Currently assuming punishment policy of the form minmax Reward_estimator
    update1, a_punish_1 = updater1(V, params1, params2, lr, 1, defect2, R_opponent_1)
    update2, a_punish_2 = updater2(V, params1, params2, lr, 2, defect1, R_opponent_2)

    return update1, update2, a_punish_1, a_punish_2

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
            **kwargs,  # these are forwarded to the parameters-to-outcomes function
            ):
    self.pr_CC_log = np.empty(n_epochs)
    self.pr_DD_log = np.empty(n_epochs)
    self.payoffs1_log = np.empty(n_epochs)
    self.payoffs2_log = np.empty(n_epochs)
    self.reward_history = np.array([])
    self.action_history = np.array([])
    self.state_history = np.array([])
    self.ipw_history = np.array([])
    self.opponent_reward_estimates_1 = np.zeros((2, 2, 2))  # Agent 2 reward at (state_1, a_2, a_1)
    self.opponent_reward_estimates_2 = np.zeros((2, 2, 2))  # Agent 1 reward at (state_2, a_1, a_2)
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

      # Observe rewards and actions
      r1, r2, a1, a2, ipw_1, ipw_2 = self.outcomes(params1, params2, self.current_state[0], self.current_state[1])
      if self.a_punish_1 is not None: # If decided to punish on the last turn, replace with punishment action
        a1 = self.a_punish_1
      if self.a_punish_2 is not None:
        a2 = self.a_punish_2
      self.state_history.append(self.current_state)
      self.reward_history.append((r1, r2))
      self.action_history.append((a1, a2))
      self.ipw_history.append((ipw_1, ipw_2))
      self.opponent_reward_estimates_1[self.current_state[0], a2, a1] += (r2 -
                                                                          self.opponent_reward_estimates_1[self.current_state[0],
                                                                                                           a2, a1]) / (i + 1)
      self.opponent_reward_estimates_2[self.current_state[1], a1, a2] += (r1 -
                                                                          self.opponent_reward_estimates_2[self.current_state[1],
                                                                                                           a1, a2]) / (i + 1)
      self.current_state = (a2, a1) # Agent i's state is agent -i's previous action

      # ToDo: make sure parameter -> action mapping is consistent!
      probs1 = T.sigmoid(params1)
      probs2 = T.sigmoid(params2)
      self.pr_CC_log[i] = probs1[0*(1-a2) + a2*3]*probs2[0*(1-a1) + 3*a1]
      self.pr_DD_log[i] = probs1[1*(1-a2) + a2*4]*probs2[1*(1-a1) + 4*a1]

      # Define bargaining value function estimator
      V1, V2 = self.payoffs(params1, params2, ipw_history, reward_history, action_history, state_history)
      def V(p1p2):
        p1, p2 = p1p2
        V1_, V2_ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
        return (V1_ + V2_).requires_grad_()

      self.payoffs1_log[i] = r1
      self.payoffs2_log[i] = r2
      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      # noinspection PyInterpreter

      # Define opponent reward functions for punishment policy
      def R_opponent_1(a2_, a1_):
        return self.opponent_reward_estimates_1[a2, a2_, a1_]

      def R_opponent_2(a1_, a2_):
        return self.opponent_reward_estimates_2[a1, a1_, a2_]

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
        R_opponent_1,
        R_opponent_2
      )

      # Compare to bargaining updates
      bargaining_update1, bargaining_update2 = self.bargaining_updates(lr, params1, params2, bargaining_updater, V,
                                                                       suboptimality_tolerance)
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

      # Do updates
      params1.data += update1
      params2.data += update2

    self.final_params = (params1, params2)
    if plot_learning:
      self.plot_last_learning()

  def plot_last_learning(self):
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
      plt.show()
    else:
      print("This learner has not learnt yet.")


class IPD_PG(PD_PGLearner):
  """
  IPD with policy gradient instead of exact solution.
  """

  def __init__(self, updater1=optim.gradient_ascent_minmax, updater2=optim.gradient_ascent_minmax):
    super().__init__()
    self.num_params1 = 5
    self.num_params2 = 5

  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    """

    :param ipw1:
    :param ipw2:
    :param reward_history:
    :param action_history:
    :param state_history:
    :return:
    """
    if type(params1) is np.ndarray or type(params2) is  np.ndarray:
      params1 = T.from_numpy(params1).float()
      params2 = T.from_numpy(params2).float()

    probs1 = T.sigmoid(params1)
    probs2 = T.sigmoid(params2)

    value_estimate_1 = 0.
    value_estimate_2 = 0.
    is_normalizer = 0. # For stability
    for ipw, r, a, s in zip(ipw_history, reward_history, action_history, state_history):
      # Get prob of a under probs1, probs2
      # ToDo: assuming params are in order [CC, CD, DC, DD]; check this!
      s1, s2 = s
      a1, a2 = a
      prob_a1 = probs1[(s1 + a1)*(1-s1) + (s1 + a1 + 1)*s1]
      prob_a2 = probs2[(s2 + a2)*(2-s2) + (s2 + a2 + 2)*s2]
      prob_a = prob_a1 * prob_a2

      # Update value estimate
      is_weight =  prob_a / ipw
      is_normalizer += is_weight
      value_estimate_1 += is_weight * r[0]
      value_estimate_2 += is_weight * r[1]

    return value_estimate_1, value_estimate_2


