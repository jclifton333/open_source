"""
Credit to Adrian Hutter from whom I copied some code.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad
from random import uniform
from functools import partial
from abc import ABCMeta, abstractmethod
import pdb
import optim
import random
import sys


class PDLearner(metaclass=ABCMeta):
  def __init__(self, **kwargs):
    # overwrite these in child classes; **kwargs can be used here
    self.num_params1 = -1
    self.num_params2 = -1
    # generate these when learning
    self.pr_CC_log = None
    self.pr_DD_log = None
    self.payoffs1_log = None
    self.payoffs2_log = None
    self.defect1 = False
    self.defect2 = False

  @abstractmethod
  def payoffs(self, **kwargs):
    """
    :param outcomes:
    :return:
    """
    pass
    payoffs1 = T.tensor([-1., -3., 0., -2.])
    payoffs2 = T.tensor([-1., 0., -3., -2.])
    return T.dot(outcomes, payoffs1), T.dot(outcomes, payoffs2)

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
                      ):
    if lr_opponent is None:
      lr_opponent = lr

    # Get actual updates
    update1 = updater1(V, lambda p1, p2: self.payoffs(self.outcomes(p1, p2))[0], params1, params2, 1, defect2,
                       lr)
    update2 = updater2(V, lambda p1, p2: self.payoffs(self.outcomes(p1, p2))[1], params1, params2, 2, defect1,
                       lr)

    return update1, update2

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
      outcomes = self.outcomes(params1, params2, **kwargs)
      assert T.allclose(T.sum(outcomes), T.tensor(1.), rtol=1e-3), f"Epoch {i + 1}: outcomes not normalized"
      assert (outcomes >= 0.).byte().all(), f"Epoch {i + 1}: outcomes not non-negative"
      self.pr_CC_log[i] = pCC = outcomes[0].data
      self.pr_DD_log[i] = pDD = outcomes[3].data
      V1, V2 = self.payoffs(outcomes)

      def V(p1p2):
        p1, p2 = p1p2
        V1_, V2_ = self.payoffs(self.outcomes(p1, p2))
        return (V1_ + V2_).requires_grad_()

      self.payoffs1_log[i] = V1
      self.payoffs2_log[i] = V2
      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      # noinspection PyInterpreter

      # Compare to the bargaining updates
      update1, update2 = self.gradient_ascent(
        lr,
        lr_opponent,
        params1,
        params2,
        updater1,
        updater2,
        self.defect1,
        self.defect2,
        V
      )
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


class IPD(PDLearner):
  def __init__(self, updater1=optim.gradient_ascent_minmax, updater2=optim.gradient_ascent_minmax):
    super().__init__()
    self.num_params1 = 5
    self.num_params2 = 5

  def outcomes(self, params1, params2, gamma=0.99):
    if type(params1) is np.ndarray or type(params2) is  np.ndarray:
      params1 = T.from_numpy(params1).float()
      params2 = T.from_numpy(params2).float()

    probs1 = T.sigmoid(params1)
    probs2 = T.sigmoid(params2)
    P = T.stack([
      probs1 * probs2,
      probs1 * (1 - probs2),
      (1 - probs1) * probs2,
      (1 - probs1) * (1 - probs2)
    ])
    s0 = P[:, 0]
    transition_matrix = P[:, 1:]
    infi_sum = T.inverse(T.eye(4) - gamma * transition_matrix)
    avg_state = (1 - gamma) * T.matmul(infi_sum, s0)
    return avg_state

  def payoffs(self, outcomes):
    """
    :param outcomes:
    :return:
    """
    payoffs1 = T.tensor([-1., -3., 0., -2.])
    payoffs2 = T.tensor([-1., 0., -3., -2.])
    return T.dot(outcomes, payoffs1), T.dot(outcomes, payoffs2)


class IPD_PG(PDLearner):
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









if __name__ == "__main__":
  ipd = IPD()
  ipd.learn(10, optim.gradient_ascent_minmax, optim.gradient_ascent_minmax, grad)
