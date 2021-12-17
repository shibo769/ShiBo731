#%% Part 2 
import pandas as pd
import numpy as np
import scipy.stats as stats
import os 
from math import ceil, sqrt

###### Initiating Parameters ######
os.chdir("D:\\Courses\\731\\HWs\\HW2")

data = pd.read_csv("Prices.csv")
data.set_index("Date", inplace = True)
data.index = pd.to_datetime(data.index)

K = 10 
alpha = 0.99
Lambda = 0.94
theta = 0.97
N = 50000

Vt = 1000000
MktCap = np.array([97.39, 158.20, 179.01, 417.97])
w = MktCap / sum(MktCap)

EWMAmu = [np.zeros((4,1))]
EWMAcov = [np.zeros((4,4))]

X = np.log(data).diff().dropna()
print("There are " + str(len(X)) + " days.")
#%% Historical method    
for n in range(len(X)): 
    EWMAmu.append((Lambda * EWMAmu[-1]).reshape(4,1) + np.array((1-Lambda) * X.iloc[n-1, :]).reshape(4,1))
    temp = np.array(np.array(X.iloc[n-1,:]).reshape(4,1) - EWMAmu[-1])
    EWMAcov.append(theta * EWMAcov[-1] + (1 - theta) * temp * temp.T)

# simulation
x = np.random.multivariate_normal(EWMAmu[-1].flatten(), EWMAcov[-1], N)
# One and ten-sqrt day loss
num = ceil(N * (1-alpha) - 1)
loss = np.sort(- Vt * (np.dot(np.exp(x), w) - 1))
VaR1 = loss[-num]
VaR10sqrt = sqrt(K) * VaR1

ES1 = (1 / (N * (1-alpha))) * \
    (np.sum(loss[ceil(N * alpha) + 1:]) + (ceil(N * alpha) - N * alpha) * loss[ceil(N * alpha)])
ES10sqrt = ES1 * sqrt(K)

print("The 1-day VaR and ES are: ", VaR1, " and ", ES1)
print("The 10-day-sqrt VaR and ES are: ", VaR10sqrt, " and ", ES10sqrt)

#%% Simulation method
loss_K_temp = []
for i in range(N): 
    Xsimu = np.zeros(shape=(K, 4))
    muT = EWMAmu[-1].flatten()
    covT = EWMAcov[-1]
    for j in range(K):
        Xsimu[j] = np.random.multivariate_normal(muT, covT, 1)
        temp = ((Xsimu[j] - muT).reshape(4,1)) * ((Xsimu[j] - muT).reshape(1,4))
        covT = theta * covT + (1-theta) * temp
        muT = Lambda * muT + (1-Lambda) * Xsimu[j]
    loss_K_temp.append(-Vt * (np.prod(np.dot(w, np.exp(Xsimu).T)) - 1))
    
lossK = np.sort(np.array(loss_K_temp))

num = ceil(N * (1-alpha) - 1)
VaRsim = lossK[-num]
ESsim = (1/(N*(1-alpha)))* \
    (np.sum(lossK[ceil(N * alpha) + 1:]) + (ceil(N * alpha) - N * alpha) * lossK[ceil(N * alpha)])

print("K = 10 VaR and ES via simulation are: ", VaRsim, " and ", ESsim)