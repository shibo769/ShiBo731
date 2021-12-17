#%%
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def squared(S1, S2, t, T, r, sig1, sig2, rho):
    part1 = np.exp(r*(T-t))
    part2 = np.power(S1,2) * np.exp(np.power(sig1,2) * (T-t))
    part3 = 2*S1*S2*np.exp(sig1*sig2*rho*(T-t))
    part4 = np.power(S2,2) * np.exp(np.power(sig2,2) * (T-t))
    temp = part1 * (part2 - part3 + part4)
    return temp

def ht(S1, S2, t, T, r, sig1, sig2, rho):
    h1 = np.exp(r*(T-t)) * (2*S1*np.exp(np.power(sig1,2) * (T-t)) - 2*S2*np.exp(sig1*sig2*rho*(T-t)))
    h2 = np.exp(r*(T-t)) * (-2*S1*np.exp(sig1*sig2*rho*(T-t)) + 2*S2*np.exp(np.power(sig2,2) * (T-t)))
    return h1,h2

if __name__ == '__main__':

    r = 1.32 / 100
    mu1 = 15.475 / 100
    mu2 = 18.312 / 100
    t = 0
    sig1 = 22.14 / 100
    sig2 = 30.36 / 100
    rho = 0.237
    T = 0.25
    S1 = 158.12
    S2 = 170.33
    dt = 1 / 252
    
    X_list = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
    sig_list = [0.5, 0.75, 1.25, 1.5, 1.75, 2]
    rho_list = [-0.5, 0, 0.5]
    prob_x = [0.5, 0.75, 1, 1, 0.75, 0.5]
    prob_sig = [0.5, 1.25, 0.75, 0.75, 1.25, 0.5]
    prob_rho = [1.25, 0.5, 1.25]
    
    x_dic = {X_list[i]:prob_x[i] for i in range(0, 6)}
    sig_dic = {sig_list[i]:prob_sig[i] for i in range(0, 6)}
    rho_dic = {rho_list[i]:prob_rho[i] for i in range(0, 3)}
    ### Beginning (a) ###
    h1,h2 = ht(S1, S2, t, T, r, sig1, sig2, rho)
    V0 = squared(S1, S2, t, T, r, sig1, sig2, rho) - h1*S1 - h2*S2
    
    Vt_list = [squared(S1*np.exp(x1), S2*np.exp(x2), t+5*dt, T, r, sig1*b1, sig2*b2, rho+rr) - h1*S1*np.exp(x1) - h2*S2*np.exp(x2) for x1 in X_list for x2 in X_list for b1 in sig_list for b2 in sig_list for rr in rho_list]
    loss_list_1 = -(Vt_list-V0)
    len(Vt_list)
    ### result = 1000
    print("The worst case scenario risk measure along with the log return/volatility/rho combination is",max(loss_list_1))
    ### End (a) ###
    
    ### Beginning (b) ###
    wtd_list = [(x_dic[x1]*x_dic[x2]*sig_dic[b1]*sig_dic[b2]*rho_dic[rr]) for x1 in X_list for x2 in X_list for b1 in sig_list for b2 in sig_list for rr in rho_list]
    max(wtd_list*loss_list_1)
    len(wtd_list)
    ### result = 1300
    print("The scenario weighted loss is",max(wtd_list*loss_list_1))
    

    