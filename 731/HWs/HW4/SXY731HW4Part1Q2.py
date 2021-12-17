#%%
import numpy as np
import pandas as pd
from scipy.stats import norm
from math import ceil
'''
Parameters
'''
K = 10
Lambda = 0.97
theta = 0.97
Vt = 1000000
W = np.array([448.77/(448.77+575.11), 575.11/(448.77+575.11)])
positions = Vt * W
alpha = 0.95
delta = 10
M = 100
N = 50000
'''
Setting
'''
df = pd.read_csv("MSFT_AAPL_Log_Returns.csv")
df.columns = ["Date","MSFT", "AAPL"]
df = df.set_index("Date")
mu_msft_init = df['MSFT'][:M].mean()
mu_aapl_init = df['AAPL'][:M].mean()
cov_initial = np.cov(df['MSFT'][:M].values, df['AAPL'][:M].values)
mu0 = np.matrix([mu_msft_init, mu_aapl_init]).T
sig0 = np.matrix(cov_initial)
msft = df["MSFT"][M:].values
aapl = df["AAPL"][M:].values
data = np.matrix([msft, aapl]).T

#%%
'''
EWMA
'''

mu_ewma = mu0
sig_ewma = sig0
for i in range(M, len(msft)):
    sig_ewma = Lambda*sig_ewma + (1-Lambda)*(data[i].T-mu_ewma)*(data[i]-mu_ewma.T)
    mu_ewma = Lambda*mu_ewma + (1-Lambda)*data[i].T
    
'''
VaR and Capital Charge
'''

positions = np.matrix(positions).T
VaR_L_linear = -np.dot(positions.T,mu_ewma) + np.sqrt(positions.T*sig_ewma*positions)*norm.ppf(alpha)
VaR_L_K_linear = VaR_L_linear * np.sqrt(K)
cap_charge = VaR_L_K_linear * 3


print("The Initial one day VaR:",VaR_L_linear)
print("The Initial 10 day VaR:",VaR_L_K_linear)
print("3x Initial 10 day VaR (capital charge):",cap_charge)
#%%
'''
Extreme Case
'''

losses = []
lossAll = np.zeros(N)
for l in range(0, N):
    ###
    mu = np.zeros([K,2]) 
    sig = np.zeros([K,2,2])
    mu[0] = mu_ewma[1]
    sig[0] = sig_ewma[1,1]
    X = np.zeros([K,2])

    x2_new = mu[0][1] - 5 * np.sqrt(sig[0][1][1]) 
    rho = sig[0][0][1] / (np.sqrt(sig[0][0][0]) * np.sqrt(sig[0][1][1]))
    mu_bar = mu[0][0] + (rho * np.sqrt(sig[0][0][0]) / np.sqrt(sig[0][1][1])) * (x2_new - mu[0][1])
    sig_bar = sig[0][0][0] * (1 - rho**2)
    x1_new = mu_bar - 5 * np.sqrt(sig_bar)
    X_new = np.array([x1_new,x2_new])
    mu[1] = Lambda * mu[0] + (1 - Lambda) * X_new
    sig[1] = theta * sig[0] + (1 - theta) * np.array(X_new - mu[0]).reshape(1,2) * np.array(X_new - mu[0]).reshape(2,1)
    
    X[0] = X_new
    X[1] = np.random.multivariate_normal(mu[1],sig[1],1)
    for i in range(2,K):
        mu[i] = Lambda * mu[i-1] + (1 - Lambda) * X[i-1]
        sig[i] = theta * sig[i-1] + (1 - theta) * np.array(X[i-1] - mu[i-1]).reshape(1,2) * np.array(X[i-1] - mu[i-1]).reshape(2,1)
        X_sim = np.random.multivariate_normal(mu[i],sig[i],1)
        X[i] = X_sim
    loss_lin = np.dot(np.array(positions).flatten(),np.array(X.sum(axis = 0)))
    lossAll[l] = -loss_lin

losses = lossAll
average_K_loss = np.mean(losses)
print("(Shocked) The average K-day portfolio loss:", average_K_loss)


nn = ceil(N * alpha)
kday_var = np.sort(np.array(losses).flatten())[nn]
print("(Shocked) The Estimate of the K-day VaRα:",kday_var)


v1 = VaR_L_K_linear.A[0]
v2 = cap_charge.A[0]
exceed1 = sum(np.where(losses > v1, 1, 0)) * 100 / N
print("(Shocked) The frequency with which the losses exceeded the initial K day VaR is", exceed1)


exceed2 = sum(np.where(losses > v2, 1, 0)) * 100 / N
print("(Shocked) The frequency with which the losses exceeded the regulatory capital ", exceed2)