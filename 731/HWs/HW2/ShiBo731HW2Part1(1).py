#%% 1. (a)
import numpy as np
import pandas as pd
import os
from math import ceil
import matplotlib.pyplot as plt
from scipy.stats import norm
# https://pythonforundergradengineers.com/unicode-characters-in-python.html
os.chdir("D:\\Courses\\731\\HWs\\HW2")

data = pd.read_csv("StockData.csv").set_index("Date")
data.index = pd.to_datetime(data.index)
print("Shape: ", data.shape)
X = np.log(data).diff().dropna()

λ = θ = 0.97
# the market capitalizations of Microsoft,
# Apple and Google were 2.269, 2.510 and 1.940 trillion dollars respectively
Cap = np.array([2.269, 2.510, 1.940])
ω_M = Cap[0] / np.sum(Cap)
ω_A = Cap[1] / np.sum(Cap)
ω_G = Cap[2] / np.sum(Cap)
ω = np.array([ω_M, ω_A, ω_G])

μ_EWMA = pd.DataFrame(index = data.index, columns = data.columns)
μ_EWMA.iloc[0] = np.mean(X) # set initial value as mean of Log return

Σ_EWMA = pd.DataFrame(index = data.index, columns = data.columns)
Σ_EWMA = [X.cov().values] # set initial value as cov of Log return

for i in range(len(X)):
    μ_EWMA.iloc[i+1] = λ * μ_EWMA.iloc[i] + (1-λ) * X.iloc[i]
    temp = np.array(X.iloc[i] - μ_EWMA.iloc[i]).reshape(3, 1)
    Σ_EWMA += [(θ * Σ_EWMA[i] + (1-θ) * temp * temp.T)]
#%% (b) i
# Empirical VaR Log return
port_size = 1000000
α = 0.95
π = X['Microsoft'] * ω_M + X['Apple'] * ω_A + X['Google'] * ω_G
Emp_VaR = π.quantile(α) * port_size

# Emprical 
n = ceil(len(X) * α - 1)
Full_loss = -port_size * np.dot((np.exp(X) - 1), ω)
Full_VaR = np.sort(Full_loss)[n]
Linear_loss = -port_size * np.dot(X, ω)
Linear_VaR = np.sort(Linear_loss)[n]
Quad_loss = -port_size * np.dot(ω, (X + 1/2 * X**2).T)
Quad_VaR = np.sort(Quad_loss)[n]

print("Empirical VaR of log return is: " + "$", Emp_VaR, \
      "\nThe Empirical VaR of full loss is:", Full_VaR, \
      "\nThe Empirical VaR of 1st approx is:", Linear_VaR, \
      "\nThe Empirical VaR of 2nd approx is:", Quad_VaR)
#%% (b) ii EWMA
N = 100000 

ewma_mu = μ_EWMA.iloc[-1]
ewma_sigma = Σ_EWMA[-1]

x = np.random.multivariate_normal(np.array(ewma_mu).astype(float), ewma_sigma, N)
num = ceil(N * 0.05 - 1)
Full_simu_ewma = sorted(-port_size * np.dot((np.exp(x) - 1), ω))[-num]
Quad_simu_ewma = sorted(-port_size * np.dot(ω, (x + 1/2 * x**2).T))[-num]

print("By using the simulation with multi-normal-EWMA method, we have,")
print("Simu Full loss: ", Full_simu_ewma)
print("Simu quad loss: ", Quad_simu_ewma)
#%% (b) ii standard estimates
stand_mu = np.mean(X)
stand_sigma = X.cov()

x = np.random.multivariate_normal(stand_mu, stand_sigma, N)
num = ceil(N * 0.05 - 1)
Full_simu = sorted(-port_size * np.dot((np.exp(x) - 1), ω))[-num]
Linear_simu_not = -ω.T.dot(np.mean(X)) + np.sqrt(np.dot(np.dot(ω.T, X.cov()), ω)) * norm.ppf(α) * port_size
Quad_simu = sorted(-port_size * np.dot(ω, (x + 1/2 * x**2).T))[-num]

print("By using the simulation with standard estimates method, we have,")
print("Simu Full loss: ", Full_simu)
print("linear loss: ", Linear_simu_not)
print("Simu quad loss: ", Quad_simu) 
