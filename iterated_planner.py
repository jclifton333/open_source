"""
Simulation optimization for finding bargaining policies (rather than bargaining updates).
"""
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.autograd import grad, set_detect_anomaly
import optim
from abc import ABCMeta, abstractmethod
from scipy.special import expit
from scipy.stats import norm
import pdb

set_detect_anomaly(True)


def defection_test(player_1_reward_history_, player_1_cooperative_reward_mean, cutoff):
  """

  :param player_1_reward_history_: Player 1's observed rewards
  :param player_1_cooperative_reward_mean: Expected value under cooperation for player 1
  :param cutoff:
  :return:
  """
  player_2_test_statistic = (np.mean(player_1_reward_history_) - player_1_cooperative_reward_mean) / \
                             np.std(player_1_reward_history_ / len(player_1_reward_history_))
  player_2_null_prob = norm.cdf(player_2_test_statistic)
  return player_2_null_prob < cutoff


class IteratedGameLearner(metaclass=ABCMeta):
  def __init__(self, payoffs1=np.array([[-1., -3.], [0., -2.]]), payoffs2=np.array([[-1., -3.], [0., -2.]])):
    """
    Generic API from which classes for learning policies with policy gradient as well as those for learning switching
    policies inherit.
    """
    # overwrite these in child classes; **kwargs can be used here
    self.num_params1 = -1
    self.num_params2 = -1
    # generate these when learning
    self.pr_CC_log = None
    self.pr_DD_log = None
    self.payoffs1 = payoffs1
    if payoffs2 is None: # If no payoffs2 provided, assume zero-sum.
      self.payoffs2 = -payoffs1
      self.zero_sum = True
    else:
      self.payoffs2 = payoffs2
      self.zero_sum = False
    self.payoffs1_log = None
    self.payoffs2_log = None
    self.reward_history = None
    self.action_history = None
    self.state_history = None
    self.ipw_history = None

  @staticmethod
  def actions_from_params(params1, params2, s1, s2):
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
    # ToDo: Assuming variance = 0.1!
    # Draw rewards
    mu1 = self.payoffs1[a1, a2]
    mu2 = self.payoffs2[a2, a1]
    r1 = np.random.normal(mu1, 0.1)
    r2 = np.random.normal(mu2, 0.1)
    return r1, r2

  @abstractmethod
  def learn(self,
            label=0,
            lr=0.1,
            updater1=optim.vanilla_gradient,
            updater2=optim.vanilla_gradient,
            std=0.01,
            n_epochs=2000,
            n_print_every=None,
            init_params1=None,
            init_params2=None,
            plot_learning=True
            ):
    pass

  def initialize_observation_histories(self, n_epochs):
    # Initialize data for learning
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

  def update_histories(self, r1, r2, a1, a2, ipw):
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
    self.current_state = (a2, a1)  # Agent i's state is agent -i's previous action

  def roll_out_switching_policy(self, n_epochs, number_of_punish_periods, cutoff):
    self.initialize_observation_histories(n_epochs)
    for i in range(n_epochs):
      # ToDo: need to reset histories at each epoch
      player2_has_defected = False  # Track whether a defection has been detected
      time_since_defection = 0
      print(i)
      for _ in range(self.time_horizon):
        if player2_has_defected:
          # ToDo: assuming player 2 continues to follow default policy after defection is detected.
          params1 = self.punishment_params1
          params2 = self.default_params2
        else:
          params1 = self.default_params1
          params2 = self.default_params2
        # ToDo: actions_from_params currently only takes tensors
        a1, a2, ipw = self.actions_from_params(self.punishment_params1, self.default_params2,
                                               self.current_state[0], self.current_state[1])
        r1, r2 = self.outcomes(a1, a2)

        # Update histories
        self.update_histories(r1, r2, a1, a2, None)
        probs1 = T.sigmoid(params1[(2 * a2):(2 * a2 + 2)])
        probs2 = T.sigmoid(params2[(2 * a1):(2 * a1 + 2)])
        probs1 /= T.sum(probs1)
        probs2 /= T.sum(probs2)
        self.pr_CC_log[i] = probs1[0] * probs2[0]
        self.pr_DD_log[i] = probs1[1] * probs2[1]

        self.payoffs1_log[i] = self.payoffs1[a1, a2]
        self.payoffs2_log[i] = self.payoffs2[a2, a1]

        # Test for defections
        # ToDo: introduce parameter for number of punish rounds
        player2_has_defected = defection_test(self.payoffs1_log, self.cooperative_payoffs1, cutoff)
        if player2_has_defected and (time_since_defection == number_of_punish_periods):
          player2_has_defected = False
          time_since_defection = 0
        elif player2_has_defected and (time_since_defection != number_of_punish_periods):
          time_since_defection += 1
    return np.mean(self.payoffs1_log)



class IteratedGamePGLearner(IteratedGameLearner):
  def __init__(self, joint_optimization=True,
               payoffs1=np.array([[-1., -3.], [0., -2.]]), payoffs2=np.array([[-1., -3.], [0., -2.]])):
    """
    Default payoffs are PD.

    :param payoffs1:
    :param payoffs2:
    """
    super().__init__(payoffs1=payoffs1, payoffs2=payoffs2)
    self.joint_optimization = joint_optimization

  @abstractmethod
  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    pass

  @staticmethod
  def gradient_ascent(lr,
                      params1_,
                      params2_,
                      updater1,
                      V_1,
                      updater2=None,
                      V_2=None):

    if V_2 is None:  # If only one value function is passed, assume joint optimization
      update1, update2 = updater1(V_1, params1_, params2_, lr)
    else:
      update1, _ = updater1(V_1, params1_, params2_, lr)
      _, update2 = updater2(V_2, params2_, params2_, lr)

    return update1, update2

  def learn(self,
            label=0,
            lr=0.1,
            updater1=optim.vanilla_gradient,
            updater2=optim.vanilla_gradient,
            std=0.01,
            n_epochs=2000,
            n_print_every=None,
            init_params1=None,
            init_params2=None,
            plot_learning=True
            ):
    self.initialize_observation_histories(n_epochs)
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

    # Learn
    for i in range(n_epochs):
      print(i)

      # Take action and observe rewards
      a1, a2, ipw = self.actions_from_params(params1, params2, self.current_state[0], self.current_state[1])
      r1, r2 = self.outcomes(a1, a2)

      self.update_histories(r1, r2, a1, a2, ipw)
      probs1 = T.sigmoid(params1[(2*a2):(2*a2 + 2)]).detach().numpy()
      probs2 = T.sigmoid(params2[(2*a1):(2*a1 + 2)]).detach().numpy()
      probs1 /= np.sum(probs1)
      probs2 /= np.sum(probs2)
      self.pr_CC_log[i] = probs1[0]*probs2[0]
      self.pr_DD_log[i] = probs1[1]*probs2[1]

      # Get gradient(s) of value function(s)
      # ToDo: this can probably be improved
      if self.joint_optimization:
        def V_1(p1p2):
          p1, p2 = p1p2
          V_1_ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
          return V_1_.requires_grad_()
        V_2 = None
      else:
        def V_1(p1p2):
          p1, p2 = p1p2
          V_1_, _ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
          return V_1_.requires_grad_()

        def V_2(p1p2):
          p1, p2 = p1p2
          _, V_2_ = self.payoffs(p1, p2, self.ipw_history, self.reward_history, self.action_history, self.state_history)
          return V_2_.requires_grad_()

      self.payoffs1_log[i] = self.payoffs1[a1, a2]
      self.payoffs2_log[i] = self.payoffs2[a2, a1]

      if n_print_every and i % n_print_every == 0:
        print(f"Epoch {i + 1} of {n_epochs}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
      # noinspection PyInterpreter

      # Get each agent's update
      update1, update2 = self.gradient_ascent(
        lr,
        params1,
        params2,
        updater1,
        V_1,
        updater2=updater2,
        V_2=V_2
      )

      # Do updates
      params1.data += update1
      params2.data += update2

    self.final_params = (params1, params2)
    if plot_learning:
      self.plot_last_learning(label)
    return {'pr_CC':self.pr_CC_log, 'pr_DD': self.pr_DD_log, 'payoffs1': self.payoffs1_log,
            'payoffs2': self.payoffs2_log, 'param1': self.final_params[0], 'param2': self.final_params[1]}

  def learn_multi_rep(self,
                      label,
                      n_rep,
                      lr,
                      updater1,
                      updater2,
                      bargaining_updater,
                      std=0.01,
                      n_epochs=2000,
                      n_print_every=None,
                      init_params1=None,
                      init_params2=None,
                      plot_learning=True,
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
                           std=std,
                           n_epochs=n_epochs,
                           n_print_every=n_print_every,
                           init_params1=init_params1,
                           init_params2=init_params2,
                           plot_learning=False,
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

  @staticmethod
  def value_estimates(params1, params2, ipw_history, reward_history, action_history, state_history):
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


class MaxMinSolver(IteratedGamePGLearner):
  # ToDo: Using naive policy gradient!
  def __init__(self, payoffs1=np.array([[-1., -3.], [0., -2.]])):
    super().__init__(joint_optimization=False, payoffs1=payoffs1)
    self.num_params1 = 4
    self.num_params2 = 4

  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    return self.value_estimates(params1, params2, ipw_history, reward_history, action_history, state_history)


class NashBargainingSolver(IteratedGamePGLearner):
  def __init__(self,
               disagreement_value_1,
               disagreement_value_2,
               payoffs1=np.array([[-1., -3.], [0., -2.]]),
               payoffs2=np.array([[-1., -3.], [0., -2.]])):
    super().__init__(joint_optimization=True, payoffs1=payoffs1, payoffs2=payoffs2)
    self.disagreement_value_1 = disagreement_value_1
    self.disagreement_value_2 = disagreement_value_2
    self.num_params1 = 4
    self.num_params2 = 4

  def payoffs(self, params1, params2, ipw_history, reward_history, action_history, state_history):
    value_estimate_1_, value_estimate_2_ = \
      self.value_estimates(params1, params2, ipw_history, reward_history, action_history, state_history)
    return T.log(value_estimate_1_ - self.disagreement_value_1) + \
        T.log(value_estimate_2_ - self.disagreement_value_2)


class SwitchingPolicyLearner(IteratedGameLearner):
  """
  Learn policy for detecting defections and switching to punishment policy.
  """
  def __init__(self,
               cooperative_payoffs1,
               punishment_params1,
               default_params1,
               default_params2,
               payoffs1=np.array([[-1., -3.], [0., -2.]]),
               payoffs2=np.array([[-1., -3.], [0., -2.]]),
               time_horizon=50):

    super().__init__(payoffs1=payoffs1, payoffs2=payoffs2)
    self.cooperative_payoffs1 = cooperative_payoffs1
    self.default_params1 = default_params1
    self.default_params2 = default_params2
    self.punishment_params1 = punishment_params1  # Parameters for punishment policy
    # ToDo: note that other learning algs don't have time horizon; each epoch impliclty has time_horizon=1
    self.time_horizon = time_horizon

  def learn(self,
            n_epochs_per_candidate_cutoff=1,
            label=0,
            lr=0.1,
            updater1=optim.vanilla_gradient,
            updater2=optim.vanilla_gradient,
            std=0.01,
            n_epochs=2000,
            n_print_every=None,
            init_params1=None,
            init_params2=None,
            plot_learning=True
            ):
    number_of_punish_periods = 10

    # Learn by naive search over grid of cutoffs
    cutoff_grid = np.linspace(0.05, 0.5, 20)
    best_cutoff = None
    # ToDo: value being optimized should be an argument, since we need to optimize both players' rewards
    best_value = -float('inf')
    for cutoff in cutoff_grid:
      value_at_cutoff = self.roll_out_switching_policy(n_epochs_per_candidate_cutoff, number_of_punish_periods, cutoff)
        # if n_print_every and i % n_print_every == 0:
        #   print(f"Epoch {i + 1} of {int(n_epochs_per_candidate_cutoff*len(cutoff_grid))}; payoffs:\t{V1:.2f}\t{V2:.2f};\tPr[CC]:\t{pCC:.2f};\tPr[DD]:\t{pDD:.2f}")
        # # noinspection PyInterpreter
      if value_at_cutoff > best_value:
        best_value = value_at_cutoff
        best_cutoff = cutoff

    return best_cutoff


def learn_switching_policies(payoffs_1, payoffs_2, n_max_min_epochs=1000, n_nbs_epochs=1000):
  """
  Learn Nash bargaining soln for minimax disagreement, then policy for switching
  to punishment.

  :param payoffs_1:
  :param payoffs_2:
  :return:
  """
  LOOK_BACK_FOR_POLICY_EVALUATION = 10  # Amount of steps to average from end of learning to estimate policy value

  # Get disagreement values
  max_min_1 = MaxMinSolver(payoffs_1)
  max_min_1_res = max_min_1.learn(n_epochs=n_max_min_epochs)
  d1 = np.mean(max_min_1_res['payoffs1'][-LOOK_BACK_FOR_POLICY_EVALUATION:])
  max_min_2 = MaxMinSolver(payoffs_2)
  max_min_2_res = max_min_2.learn(n_epochs=n_max_min_epochs)
  d2 = np.mean(max_min_2_res['payoffs1'][-LOOK_BACK_FOR_POLICY_EVALUATION:])

  # Get Nash bargaining solution
  nbs = NashBargainingSolver(d1, d2, payoffs1=payoffs_1, payoffs2=payoffs_2)
  nbs_res = nbs.learn(n_epochs=n_nbs_epochs)

  # Get switching policies
  # ToDo: here the switching policy learner is learning against the Nash bargaining policy, whereas it should
  # ToDo: be learning against a defecting policy.
  switcher1 = SwitchingPolicyLearner(np.mean(nbs_res['payoffs1'][-LOOK_BACK_FOR_POLICY_EVALUATION:]),
                                     max_min_2_res['param2'],
                                     nbs_res['param1'],
                                     nbs_res['param2'])
  cutoff1 = switcher1.learn()
  switcher2 = SwitchingPolicyLearner(np.mean(nbs_res['payoffs2'][-LOOK_BACK_FOR_POLICY_EVALUATION:]),
                                     max_min_1_res['param2'],
                                     nbs_res['param2'],
                                     nbs_res['param1'])
  cutoff2 = switcher2.learn()
  return cutoff1, cutoff2


if __name__ == "__main__":
  pd_payoffs = np.array([[-1., -3.], [0., -2.]])
  print(learn_switching_policies(pd_payoffs, pd_payoffs))

