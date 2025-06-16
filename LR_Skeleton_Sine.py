#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:55:44 2025

@author: shoaibazmat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1)  # For reproducibility

#Actual Sine
xt = np.linspace(0, 1, 100)
YT = np.sin(2* np.pi * xt) 

#Noisy Sine Data
x1 = np.linspace(0, 1, 10)
Y = np.sin(2* np.pi * x1) + np.random.normal(0,0.20,size=x1.size)


#TODO

#write polynomial regression code to find h_theta

plt.scatter(x1, Y, label='Sample Data')
plt.plot(xt, YT, label='Actual Curve')
#plt.plot(x1,h_theta,label='Predicted Curve')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine')
plt.legend()
plt.show()