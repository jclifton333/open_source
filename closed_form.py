import numpy as np
import numdifftools as nd
import pdb
from scipy.special import expit


def get_transition_matrix(p_cc_1, p_dc_1, p_cc_2, p_dc_2):
  """
  States are ordered
  CC, CD, DC, DD.

  :param p_cc_1:
  :param p_dc_1:
  :param p_cc_2:
  :param p_dc_2:
  :return:
  """
  from_cc = np.array([p_cc_1*p_cc_2, p_cc_1*(1-p_cc_2), (1-p_cc_1)*p_cc_2, (1-p_cc_1)*(1-p_cc_2)])
  from_cd = np.array([p_dc_1*p_cc_2, p_dc_1*(1-p_cc_2), (1-p_dc_1)*p_cc_2, (1-p_dc_1)*(1-p_cc_2)])
  from_dc = np.array([p_cc_1*p_dc_2, p_cc_1*(1-p_dc_2), (1-p_cc_1)*p_dc_2, (1-p_cc_1)*(1-p_dc_2)])
  from_dd = np.array([p_dc_1*p_dc_2, p_dc_1*(1-p_dc_2), (1-p_dc_1)*p_dc_2, (1-p_dc_1)*(1-p_dc_2)])

  transition_matrix = np.array([from_cc, from_cd, from_dc, from_dd])
  return transition_matrix


def get_game_hessian(payoffs_1, payoffs_2, gamma=0.9):
  def value_function_1(p):
    true_p = expit(p)
    p_cc_1, p_dc_1, p_cc_2, p_dc_2 = true_p[0], true_p[1], true_p[2], true_p[3]
    transition_matrix = get_transition_matrix(p_cc_1, p_dc_1, p_cc_2, p_dc_2)
    v1 = np.dot(np.linalg.inv(np.eye(4) - gamma*transition_matrix), payoffs_1)
    return v1[0]

  def value_function_2(p):
    true_p = expit(p)
    p_cc_1, p_dc_1, p_cc_2, p_dc_2 = true_p[0], true_p[1], true_p[2], true_p[3]
    transition_matrix = get_transition_matrix(p_cc_1, p_dc_1, p_cc_2, p_dc_2)
    v2 = np.dot(np.linalg.inv(np.eye(4) - gamma*transition_matrix), payoffs_2)
    return v2[0]

  hess_1 = nd.Hessian(value_function_1)
  hess_2 = nd.Hessian(value_function_2)
  hess = lambda p: np.vstack((hess_1(p)[:2, :], hess_2(p)[2:, :]))
  return hess


if __name__ == "__main__":
  # ToDo: Somehow make parameter space unbounded? (this explains why eigenvals aren't all positive)
  stag_payoffs_1 = np.array([2, -1, 1, 1])
  stag_payoffs_2 = np.array([2, 1, -1, 1])
  pd_payoffs_1 = np.array([1, -1, 2, -3])
  pd_payoffs_2 = np.array([1, 2, -1, -3])
  chicken_payoffs_1 = np.array([0, -1, 1, -5])
  chicken_payoffs_2 = np.array([0, 1, -1, -5])
  dummy_payoffs_1 = np.array([1, -1, 1, -1])
  dummy_payoffs_2 = np.array([-1, 1, -1, 1])
  # stag_grad = get_game_hessian(stag_payoffs_1,stag_payoffs_2)
  # print(np.linalg.eig(stag_grad(np.array([1, 0, 1, 0])))[0])
  # pd_grad = get_game_hessian(pd_payoffs_1,pd_payoffs_2)
  # print(np.linalg.eig(pd_grad(np.array([1, 1, 1, 1])))[0])
  # chicken_grad = get_game_hessian(chicken_payoffs_1,chicken_payoffs_2)
  # print(np.linalg.eig(chicken_grad(np.array([1, 1, 0, 0])))[0])
  dummy_grad = get_game_hessian(dummy_payoffs_1,dummy_payoffs_2)
  print(np.linalg.eig(dummy_grad(np.array([1, 1, 1, 1])))[0])

