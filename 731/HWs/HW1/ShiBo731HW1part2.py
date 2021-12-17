import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

µ = 0.16905 # return
σ = 0.4907 # volatility
r = 0.0011888 # interest rate
t = 0 # time
T = 0.291667 # time period
Δ = 10/252 
S = 152.51 # S0
K = 170 # K
M = 100 # position
N = 100000 # # of simulations

d1 = (np.log(S/K) + (r + 0.5*σ**2)*T)/(σ * np.sqrt(T))
d2 = (np.log(S/K) + (r - 0.5*σ**2)*T)/(σ * np.sqrt(T))

delta = norm.cdf(d1) - 1
theta = -σ / (2 * T**0.5) * S * norm.pdf(d1) + K * r * np.exp(-r*T) * (1-norm.cdf(d2))
gamma = norm.pdf(d1)/(S * σ * T**0.5)

p = K * np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

draws = np.random.normal((µ - σ**2 / 2) * Δ, (σ ** 2 * Δ) ** 0.5, N)

new_d1 = (np.log(S * np.exp(draws) / K) + (r + 0.5 * σ**2) * (T - Δ))/(σ * np.sqrt(T - Δ))
new_d2 = (np.log(S * np.exp(draws) / K) + (r - 0.5 * σ**2) * (T - Δ))/(σ * np.sqrt(T - Δ))

P = K * np.exp(-r*(T - Δ))*norm.cdf(-new_d2) - S * np.exp(draws) * norm.cdf(-new_d1)

FullLoss = - M * (delta * S *(np.exp(draws) - 1) - (P - p))
LinearLoss = - M * theta * Δ 
QuadLoss = - LinearLoss - (0.5 * M * (delta * S - gamma * S ** 2) * draws ** 2)

sns.distplot(FullLoss, color="b", bins = 100)
plt.title("Loss density plot for Full Loss")
plt.xlim([-100,500])
plt.show()

sns.distplot(LinearLoss, color="b", bins = 100)
plt.title("Loss density plot for 1st approximation")
plt.show()

sns.distplot(QuadLoss, color="b", bins = 100)
plt.title("Loss density plot for 2nd approximation")
plt.xlim([-100,500])
plt.show()