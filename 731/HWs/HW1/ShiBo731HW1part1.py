# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 00:06:30 2021

@author: boshi
"""

#%% Part I 
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

delta = 1/252
n = 100
m = 100
lambda1 = 0.94
lambda2 = 0.97

data = pd.read_csv("D:\\Courses\\731\\HWs\\HW1\\SPData.csv")
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

log_ret = np.log(data['Closing Price']).diff().dropna()
#%% Moving Average and EWMA
MA = pd.Series([0] * (len(log_ret) - n))
for i in range(len(log_ret)-n):
    MA.iloc[i] = log_ret.iloc[i: i+n].var()/delta

InitVol = np.var(log_ret[0:n])

EWMA94 = pd.Series([0] * (len(log_ret) - n))
EWMA97 = pd.Series([0] * (len(log_ret) - n))
EWMA94.iloc[0] = InitVol
EWMA97.iloc[0] = InitVol

for i in range(len(log_ret) - n):
    EWMA94.iloc[i] = lambda1*EWMA94.iloc[i-1]+(1-lambda1)*(log_ret[n:]**2).iloc[i-1]
    EWMA97.iloc[i] = lambda2*EWMA97.iloc[i-1]+(1-lambda2)*(log_ret[n:]**2).iloc[i-1]

EWMA94 = EWMA94 / delta
EWMA97 = EWMA97 / delta

Res_df = pd.DataFrame({"MA" : np.array(MA), "EWMA94" : np.array(EWMA94), "EWMA97" : np.array(EWMA97)},
                      index = log_ret.index[n:])

plt.figure(dpi = 120, figsize = (12,8))
plt.plot(np.sqrt(Res_df))
plt.legend(Res_df.columns, prop={'size': 20})
plt.title("Comparsion Between MA and EWMA", fontsize = 20)
plt.xlabel("Date", fontsize = 20)
plt.xticks(rotation = 45)
plt.ylabel("Volatility", fontsize = 20)
#%% GARCH
Garch_df = log_ret.loc['2019-03-01': '2020-02-28'] * np.sqrt(1/delta)
garch = arch_model(Garch_df, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='Normal')
results = garch.fit()
print(results.summary())

omega = results.params.omega
alpha1 = results.params.loc['alpha[1]']
beta1 = results.params.loc['beta[1]']
### Long term volatility
Vl = omega/(1-alpha1-beta1)
### Innovation
zt = np.random.normal(0, 1, (42, m))
### Simulation
simu_vol = []
simu_vol = [[Vl] * m]
for i in range(1, 43, 1): 
    simu_vol += [omega + simu_vol[i-1] * (alpha1*zt[i-1]**2+beta1)]

simu_df = pd.DataFrame(simu_vol)
### Plot
SubRes_df = Res_df.loc["2020-03":"2020-04"]

plt.figure(figsize=(20,10), dpi = 120)
# plt.plot(np.sqrt(simu_df))
plt.plot(np.array(np.sqrt(SubRes_df.MA)), color = 'black', linewidth = 3, label = "MA")
plt.plot(np.array(np.sqrt(SubRes_df.EWMA94)), color = 'brown', linewidth = 3, label = "EWMA94")
plt.plot(np.array(np.sqrt(SubRes_df.EWMA97)), color = 'grey', linewidth = 3, label = "EWMA97")
plt.legend(prop = {"size" : 20})
plt.title('Two-month ' + str(m) +' sample path for annualized volatility',fontsize = 25)
plt.xlabel("Date",fontsize = 20)
plt.ylabel("Annualized vol",fontsize = 20)
plt.show()