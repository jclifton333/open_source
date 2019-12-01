import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad
from random import uniform
from functools import partial
from abc import ABCMeta, abstractmethod
import random
import sys


# ToDo: Implement parent class for Lola and SOS
class SOSLearner(metaclass=ABCMeta):
  def __init__(self, payoffs1=[-1, -3, 0, -2], payoffs2=[-1, 0, -3, -2], **kwargs):
    # overwrite these in child classes; **kwargs can be used here
    self.num_params1 = -1
    self.num_params2 = -1
    # generate these when learning
    self.pr_CC_log = None
    self.pr_DD_log = None
    self.payoffs1_log = None
    self.payoffs2_log = None
    self.payoffs1 = payoffs1
    self.payoffs2 = payoffs2

  @abstractmethod
  def outcomes(self, params1, params2, **kwargs):
    """returns tensor [pr_CC, pr_CD, pr_DC, pr_DD]"""
    pass

  def payoffs(self, outcomes):
    payoffs1 = T.tensor(self.payoffs1)
    payoffs2 = T.tensor(self.payoffs2)
    return T.dot(outcomes, payoffs1), T.dot(outcomes, payoffs2)

  def gradient_ascent(self,
                      lr,
                      lr_opponent,
                      params1,
                      params2,
                      V1,
                      V2
                      ):
    # From SOS IPD experiment
    a = 0.5
    b = 0.1

    if lr_opponent is None:
      lr_opponent = lr
    dV1d1, dV1d2 = grad(V1, (params1, params2), create_graph=True)
    dV2d1, dV2d2 = grad(V2, (params1, params2), create_graph=True)
    dV1 = T.stack([dV1d1, dV1d2], 1)
    dV2 = T.stack([dV2d1, dV2d2], 1)
    dV = T.cat([dV1, dV2], 0)

    # Compute opponent Hessians
    Ho_1 = T.stack([grad(d, params1, retain_graph=True)[0] for d in dV1d2])
    Ho_top = T.cat([T.zeros(Ho_1.shape), Ho_1], 1)
    Ho_2 = T.stack([grad(d, params2, retain_graph=True)[0] for d in dV2d1])
    Ho_bottom = T.cat([Ho_2, T.zeros(Ho_2.shape)], 1)
    Ho = T.cat([Ho_top, Ho_bottom], 0)

    # xi is vector of gradients of player-specific losses wrt corresponding player-specific parameters
    xi = T.cat([dV1d1, dV2d2])
    xi_norm_sq = T.dot(xi, xi)
    # ToDo: make sure signs are correct, since SOS paper uses minimization of loss (instead of max value)
    xi_10 = T.mv((T.eye(len(xi)) - lr*Ho), xi)
    # ToDo: check dimensions in mat vec multiplication
    chi_11 = T.mv(Ho_top, T.cat([dV1d1, dV2d1], 0))
    chi_12 = T.mv(Ho_bottom, T.cat([dV2d1, dV2d2], 0))
    chi_1 = T.cat([chi_11, chi_12], 0)
    chi_dot_xi_1 = T.dot(-lr*chi_1, xi_10)
    if chi_dot_xi_1 > 0:
      p1 = 1.
    else:
      xi_10_norm_sq = T.dot(xi_10, xi_10)
      p1 = np.min((1., -a*xi_10_norm_sq / chi_dot_xi_1))
    if T.sqrt(xi_norm_sq) < b:
      p2 = xi_norm_sq
    else:
      p2 = 1.
    p = np.min((p1, p2))
    xi_1p = xi_10 - p*lr*chi_1
    params1.data -= lr*xi_1p[:len(params1)]
    params2.data -= lr*xi_1p[len(params1):]

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
      V1, V2 = self.payoffs(outcomes)
      V1, V2 = -V1, -V2
      self.payoffs1_log[i] = V1
      self.payoffs2_log[i] = V2
      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      self.gradient_ascent(
        lr,
        lr_opponent,
        params1,
        params2,
        V1,
        V2
      )

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


class LolaLearner(metaclass=ABCMeta):
  def __init__(self, payoffs1=[-1, -3, 0, -2], payoffs2=[-1, 0, -3, -2], **kwargs):
    # overwrite these in child classes; **kwargs can be used here
    self.num_params1 = -1
    self.num_params2 = -1
    # generate these when learning
    self.pr_CC_log = None
    self.pr_DD_log = None
    self.payoffs1_log = None
    self.payoffs2_log = None
    self.payoffs1 = payoffs1
    self.payoffs2 = payoffs2

  @abstractmethod
  def outcomes(self, params1, params2, **kwargs):
    """returns tensor [pr_CC, pr_CD, pr_DC, pr_DD]"""
    pass

  def payoffs(self, outcomes):
    payoffs1 = T.tensor(self.payoffs1)
    payoffs2 = T.tensor(self.payoffs2)
    return T.dot(outcomes, payoffs1), T.dot(outcomes, payoffs2)

  def gradient_ascent(self,
                      lr,
                      lr_opponent,
                      params1,
                      params2,
                      V1,
                      V2,
                      lola1,
                      lola2,
                      include_second_lola_term1,
                      include_second_lola_term2,
                      ):
    if lr_opponent is None:
      lr_opponent = lr
    dV1d1, dV1d2 = grad(V1, (params1, params2), create_graph=True)
    dV2d1, dV2d2 = grad(V2, (params1, params2), create_graph=True)

    params1.data += (lr * dV1d1).data
    if lola1:
      dV2d21 = T.stack([grad(d, params1, retain_graph=True)[0] for d in dV2d2])
      params1.data += lr * lr_opponent * T.matmul(dV1d2, dV2d21).data
      if include_second_lola_term1:
        dV1d21 = T.stack([grad(d, params1, retain_graph=True)[0] for d in dV1d2])
        params1.data += lr * lr_opponent * T.matmul(dV2d2, dV1d21).data

    params2.data += (lr * dV2d2).data
    if lola2:
      dV1d12 = T.stack([grad(d, params2, retain_graph=True)[0] for d in dV1d1])
      params2.data += lr * lr_opponent * T.matmul(dV2d1, dV1d12).data
      if include_second_lola_term2:
        dV2d12 = T.stack([grad(d, params2, retain_graph=True)[0] for d in dV2d1])
        params2.data += lr * lr_opponent * T.matmul(dV1d1, dV2d12).data

  def learn(self,
            lr,
            lr_opponent=None,
            lola1=False,
            lola2=False,
            include_second_lola_term1=False,
            include_second_lola_term2=False,
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
      V1, V2 = self.payoffs(outcomes)
      self.payoffs1_log[i] = V1
      self.payoffs2_log[i] = V2
      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      self.gradient_ascent(
        lr,
        lr_opponent,
        params1,
        params2,
        V1,
        V2,
        lola1,
        lola2,
        include_second_lola_term1,
        include_second_lola_term2,
      )

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


class IteratedSOS(SOSLearner):
  def __init__(self, payoffs1=[-1., -3., 0., -2.], payoffs2=[-1., 0., -3., -2.]):
    super().__init__(payoffs1=payoffs1, payoffs2=payoffs2)
    self.num_params1 = 5
    self.num_params2 = 5
    self.payoffs1 = payoffs1

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


if __name__ == "__main__":
  stag_payoffs1 = [2., -3., 1., 1.]
  stag_payoffs2 = [2., 1., -3., 1.]
  istag = IteratedSOS(payoffs1=stag_payoffs1, payoffs2=stag_payoffs2)
  # istag.learn(10.)
  ipd = IteratedSOS()
  np.random.seed(4)
  ipd.learn(1.)
