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
  def outcomes(self, params1, params2, **kwargs):
    """returns tensor [pr_CC, pr_CD, pr_DC, pr_DD]"""
    pass

  def payoffs(self, outcomes):
    payoffs1 = T.tensor([-1., -3., 0., -2.])
    payoffs2 = T.tensor([-1., 0., -3., -2.])
    return T.dot(outcomes, payoffs1), T.dot(outcomes, payoffs2)

  def bargaining_updates(self, lr, params1, params2, bargaining_updater, V, suboptimality_tolerance):
    """
    Get updates from bargaining updaters (in order to check whether these were actually followed.)

    :return:
    """
    # Get correct updates
    dVd1, dVd2 = bargaining_updater(V, (params1, params2), create_graph=True)
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
                      suboptimality_tolerance=0.1
                      ):
    if lr_opponent is None:
      lr_opponent = lr

    # Get actual updates
    update1 = updater1(V, (params1, params2), defect2, create_graph=True)
    update2 = updater2(V, (params1, params2), defect1, create_graph=True)

    return update1, update2

  def learn(self,
            lr,
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
      outcomes = self.outcomes(params1, params2, **kwargs)
      assert T.allclose(T.sum(outcomes), T.tensor(1.), rtol=1e-3), f"Epoch {i + 1}: outcomes not normalized"
      assert (outcomes >= 0.).byte().all(), f"Epoch {i + 1}: outcomes not non-negative"
      self.pr_CC_log[i] = pCC = outcomes[0].data
      self.pr_DD_log[i] = pDD = outcomes[3].data
      V = self.payoffs(outcomes)
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
        self.defect1,
        self.defect2,
        V
      )
      bargaining_update1, bargaining_update2 = self.bargaining_updates(lr, params1, params2, bargaining_updater, V,
                                                                       suboptimality_tolerance)
      V_bargaining = V((params1 + bargaining_update1, params2 + bargaining_update2))
      if V_bargaining < V((params1 + bargaining_update1, params2 + update2)) + suboptimality_tolerance:
        self.defect2 = False
      else:
        self.defect2 = True
      if V_bargaining < V((params1 + update1, params2 + bargaining_update2)) + suboptimality_tolerance:
        self.defect1 = False
      else:
        self.defect1 = True

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
  def __init__(self):
    super().__init__()
    self.num_params1 = 5
    self.num_params2 = 5

  def outcomes(self, params1, params2, gamma=0.99):
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