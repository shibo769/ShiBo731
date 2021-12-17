### Initialization ###
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import chi2
import matplotlib.pyplot as plt
alpha = 0.97
beta = 0.02
M = 125000
V = 1000000
data = pd.read_csv("AAPL_Data.csv").set_index("Date")
data["logret"] = np.log(data/data.shift())
data.dropna(inplace = True)
data["loss"] = -V * (np.exp(data.logret)-1)
### Unknown Sigma ###
EmpVar = data.loss.quantile(alpha, interpolation = "higher")
print("The empirical VaR is:", EmpVar)
logret = data["logret"]
StandVaR = V * (1 - np.exp(logret.mean() + logret.std() * norm.ppf(1-alpha)))
print("The Standard Estimats VaR is:", StandVaR)
n = len(logret)
SigLower = np.sqrt((n-1)/chi2.ppf(1-beta/2, df=n-1)) * logret.std()
SigUpper = np.sqrt((n-1)/chi2.ppf(beta/2, df=n-1)) * logret.std()
VaRLower = V * (1 - np.exp(logret.mean() + SigLower * norm.ppf(1-alpha)))
VaRUpper = V * (1 - np.exp(logret.mean() + SigUpper * norm.ppf(1-alpha)))
print("The standard estimator CI is :","[",VaRLower,",",VaRUpper,"].")
### Both Unknown ###
Chi2Sim = np.random.chisquare(df = n - 1, size = M)
SigSim = np.sqrt((n-1) / Chi2Sim) * logret.std()
MuSim = np.random.normal(logret.mean(), SigSim**2 / n)
VaRSim = V * (1-np.exp(MuSim + SigSim * norm.ppf(1-alpha)))
VaRSimAvg = VaRSim.mean()
print("The Ym (The Average Simulated VaR) is:", VaRSimAvg)
A = np.quantile(VaRSim, beta/2, interpolation="higher")
B = np.quantile(VaRSim, 1-beta/2, interpolation="higher")
print("The CI for the simulated average VaR is: (",A,",",B,")")