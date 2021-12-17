#%%
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import invgamma
from scipy.integrate import quad
from math import ceil, exp, sqrt
from scipy.stats import genpareto as gp
import pandas as pd
import numpy as np
#%% Setting
lam = 100
mu = 0.1
sigma = 0.4
M = 1000000
alpha0 = 0.99
#%%
### Sn ###
N_mean = lam
N_var = lam
X_mean = exp(mu+sigma**2/2)
X_var = exp(2*mu+sigma**2)*(exp(sigma**2)-1)
X_mean_sq = exp(2*mu+2*sigma**2)
X_mean_cube = exp(3*mu+9*sigma**2/2)
SN_mean = N_mean * X_mean
SN_var = N_mean * X_var + (X_mean)**2 * N_var
SN_skew = (X_mean_cube) / sqrt(lam*(X_mean_sq)**3)
### Gamma  ###
gamma_alpha = 4 * (SN_skew)**(-2)
gamma_beta = sqrt(gamma_alpha / (lam * X_mean_sq))
gamma_k = lam * X_mean - gamma_alpha / gamma_beta
### Normal approx ###
norm_approx = lambda x: norm.cdf(x, SN_mean, sqrt(SN_var))
### Gamma approx ###
gamma_approx = lambda x: gamma.cdf(x - gamma_k, gamma_alpha, scale = 1 / gamma_beta)
### Poisson simu ###
Ns = np.random.poisson(lam, M)
### Log-normal simu ###
Xs = [np.random.lognormal(mu, sigma, N) for N in Ns]
### Compound ###
SNs = sorted(np.array([np.sum(X) for X in Xs]))
### GP ###
lowbdd = np.quantile(SNs, alpha0, interpolation='higher')
highbdd = np.quantile(SNs, 0.99999, interpolation='higher')
u = np.quantile(SNs, alpha0, interpolation='higher')
SNu = pd.Series(SNs)
SN_exceed = (SNu[SNu>u] - u).values 
shape, loc, scale = gp.fit(SN_exceed, floc=0)
### cdf values
cdf_xasix = np.linspace(lowbdd, highbdd, 1000)
G_cdf = lambda x: 1 - (1 + shape*x/scale)**(-1/shape)
gp_approx = lambda x: G_cdf(x - u)*(1 - alpha0) + alpha0
### Emp ###
N_emp = np.random.poisson(lam, M)
X_emp = [np.random.lognormal(mu, sigma, N) for N in Ns]
SNs_emp = sorted(np.array([np.sum(X) for X in Xs]))
lowbdd_ind = SNs.index(lowbdd)
highbdd_ind = SNs.index(highbdd)
emp_len = len(SNs[lowbdd_ind: highbdd_ind+1])
cdf_emp = np.linspace(lowbdd_ind, highbdd_ind, emp_len) / M
#%% Plots ###
plt.plot(SNs[lowbdd_ind: highbdd_ind+1], 1 - cdf_emp)
plt.plot(cdf_xasix, 1 - norm_approx(cdf_xasix))
plt.plot(cdf_xasix, 1 - gamma_approx(cdf_xasix))
plt.plot(cdf_xasix, 1 - gp_approx(cdf_xasix))
plt.ylabel('1 - F(x)')
plt.xlabel('x')
plt.xscale("log")
plt.yscale("log")
plt.legend(['Empirical','Normal','Gamma','GP'])
plt.title("Log-log plot for Approximating Compound Poisson", fontsize = 13)
plt.show()
#%%
def gammaES(a, b, alpha):
    temp = a / b + gamma.ppf(alpha,a) * gamma.pdf(gamma.ppf(alpha, a), a)  / (b * (1-alpha))
    return temp
## GAMMA ES
seq_alpha = np.arange(alpha0, 0.99999, 0.00001)
gammaES = gammaES(gamma_alpha, gamma_beta, seq_alpha)
## NORMAL ES
normalES = np.zeros(len(seq_alpha))
for i in range(len(seq_alpha)):
    normalES[i] = sum(SNs[int(len(SNs)*seq_alpha[i]):])/(len(SNs)*(1-seq_alpha[i]))
## GP ES
GPVaR = u + scale/shape * (((1-alpha0)/(1-seq_alpha))**shape - 1)
GPES = (GPVaR + scale - shape * u) / (1-shape)
    
plt.figure()
plt.plot(seq_alpha, gammaES)
plt.plot(seq_alpha, GPES)
plt.plot(seq_alpha, normalES)
plt.xlabel("alpha")
plt.ylabel("ES")
plt.title("ES for Sn")
plt.show()