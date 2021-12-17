import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

######### a) empirical
data = pd.read_csv("SP_Prices.csv")
data.set_index("Date", inplace = True)
LOGRet = np.log(data).diff().dropna() 
window = 1010
alpha = 0.95
X = LOGRet.rolling(window = window).quantile(1-alpha).dropna() # rolling log return quantile 0.05
exceed = LOGRet - X
EMPn = (exceed["Price"]<0).sum()
print('The exceedences for empirical distribution is: ', EMPn)
plt.figure(dpi = 120)
plt.plot(np.array(-X))
plt.title("4 years Rolling VaR for Empirical method")
plt.xlabel("T")
plt.ylabel("VaR")
#%%
########## b) EWMA
Lambda = 0.97
mu = [np.mean(LOGRet.iloc[:window, 0])]
sig2 = [np.var(LOGRet.iloc[:window, 0])]
VaR = []
for i in range(window, len(LOGRet)):
    mu.append(Lambda* mu[-1] + (1 - Lambda) * LOGRet.iloc[i-1, 0])
    sig2.append(Lambda * sig2[-1] + (1-Lambda) * ((LOGRet.iloc[i-1, 0] - mu[-1]) ** 2))
    VaR.append(-(np.exp(mu[-1] + (sig2[-1] ** 0.5) * norm.ppf(1-alpha)) - 1))
exceed = np.array(-(LOGRet.iloc[window:, 0] - np.array(VaR)))
EWMAn = (exceed < 0).sum()
plt.figure(dpi = 120)
plt.plot(VaR)
plt.title("4 years Rolling VaR for EWMA method")
plt.xlabel("T")
plt.ylabel("VaR")
print('The exceedence for using EWMA method is ', EWMAn)