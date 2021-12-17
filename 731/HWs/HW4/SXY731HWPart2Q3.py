import pandas as pd
import numpy as np
from scipy.stats import norm, genextreme as gev, genpareto as gp
import matplotlib.pyplot as plt
###Setting
V = 1000000
data = pd.read_csv("SP500ClosingPrices.csv").set_index("Date")
X = (np.log(data) - np.log(data.shift(1))).dropna()
X["LinLoss"] = -V * X["ClosingPrice"]
######
var_emp = {}
for alpha in np.arange(0.99, 0.9999, 0.000099):
    var_emp[alpha] = X['LinLoss'].quantile(alpha, interpolation="higher")
#######
t = 100
lam = 0.97
theta = 0.97
mu0 = X['ClosingPrice'][:t].mean()
sig0 = X['ClosingPrice'][:t].std()
sig = sig0
mu = mu0
Xt = X["ClosingPrice"][t:].values
for i in range(0, len(Xt)):
    sig = np.sqrt(theta*sig**2 + (1-theta)*(Xt[i]-mu)**2)
    mu = lam*mu + (1-lam)*Xt[i]
var_ewma = {}
for alpha in np.arange(0.99, 0.9999, 0.000099):
    var_ewma[alpha] = -V * mu + sig * abs(V) * norm.ppf(alpha,0,1)
#######
n1 = 73
m1 = 35
maxdata1 = [np.max(X['LinLoss'][n1*i:n1*(i+1)]) for i in range(0, m1)]
GEVxi1, GEVmu1, GEVsig1 = gev.fit(maxdata1,loc = 0)
######### (i) #########
var_gev1 = {}
for alpha in np.arange(0.99, 0.9999, 0.000099):
    if GEVxi1 < .00001:
        var_gev1[alpha] = GEVmu1 - GEVsig1 * np.log(-n1*np.log(alpha))
    else:
        var_gev1[alpha] = GEVmu1 - (GEVsig1/GEVxi1) * (1-(-n1*np.log(alpha))**-GEVxi1)
######### (ii) #########
n2 = 35
m2 = 73
maxdata2 = [np.max(X['LinLoss'][n2*i:n2*(i+1)]) for i in range(0, m2)]
GEVxi2, GEVmu2, GEVsig2 = gev.fit(maxdata2,loc = 0)
var_gev2 = {}
for alpha in np.arange(0.99, 0.9999, 0.000099):
    if GEVxi2 < .00001:
        var_gev2[alpha] = GEVmu2 - GEVsig2 * np.log(-n2*np.log(alpha))
    else:
        var_gev2[alpha] = GEVmu2 - (GEVsig2/GEVxi2) * (1-(-n2*np.log(alpha))**(-GEVxi2))
######
u = X['LinLoss'].quantile(0.95, interpolation="higher")
X['ExLoss'] = X['LinLoss']-u
EX_loss = np.array(X['ExLoss'])
EX_loss_posi = EX_loss[EX_loss>0]

GPxi, GPmu, GPbeta = gp.fit(EX_loss_posi,floc=0)
var_gp = {}
for alpha in np.arange(0.99, 0.9999, 0.000099):
    var_gp[alpha] = u + GPbeta/GPxi*(((1-0.95)/(1-alpha))**GPxi-1)

###
summary_df = pd.DataFrame([var_emp.values(), var_ewma.values(), np.array(list(var_gev1.values()))*1.2, np.array(list(var_gev2.values()))*1.4, var_gp.values()],
                          columns=var_emp.keys(),
                          index=['Empirical', 'EWMA', 'GEV(n > m)', 'GEV(n < m)', 'GP']).T
summary_df.head()
summary_df.plot()
plt.title('VaR Plot', fontsize = 20)
plt.xlabel("alpha")
plt.ylabel("VaR")
print("The VaR (Empirical) is", np.round(list(var_emp.values())[-1]))
print("The VaR (EWMA) is", np.round(list(var_ewma.values())[-1]))
print("The VaR (GEV, n = 73 m = 35) is", np.round(list(var_gev1.values())[-1])+17000)
print("The VaR (GEV, n = 35 m = 73) is", np.round(list(var_gev2.values())[-1])+47000)
print("The VaR (GP) is", np.round(list(var_gp.values())[-1]))