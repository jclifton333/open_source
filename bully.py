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
"""
Coding 
  Types 
    Bluffer=0
    Tough=1 
  Actions
    Choice phase 
      Quiche=0
      Beer=1 
    Interaction phase 
      Threatener decides to threaten, or not (1, 0)
        Target decides to give in, or not (0, 1)
          Threatener attacks, or not (1, 0)
"""

