#%% Problem 2
### Initialization
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
from math import exp

def load_data(data):
    data.index = pd.to_datetime(data.index)
    dates = data.index
    X = np.log(data).diff().dropna()
    return X, dates

def f(u, gamma = 25):
    exponential_spectral = (gamma * exp(gamma * u)) / (exp(gamma) - 1)
    return norm.ppf(u) * exponential_spectral

def graphs():
    ### Graphs for VaR
    df_rel_comp_var_data = pd.DataFrame(rel_comp_var_data[1: ], index=date_vals, columns=data.columns)
    plt.figure(figsize = (12,8), dpi = 120)
    plt.plot(df_rel_comp_var_data)
    plt.xlabel("Dates", fontsize = 12)
    plt.xticks(rotation = 45)
    plt.ylabel("% VaR Contribution", fontsize = 12)
    plt.title("Time-series VaR Contribution Plot", fontsize = 15)
    plt.legend(df_rel_comp_var_data.columns)
    plt.show()
    
    ### Contribution of Portfolio Variance
    df_pct_variance_data = pd.DataFrame(pct_variance_data[1: ], index=date_vals, columns=data.columns)
    plt.figure(figsize = (12,8), dpi = 120)
    plt.plot(df_pct_variance_data)
    plt.xlabel("Dates", fontsize = 12)
    plt.xticks(rotation = 45)
    plt.ylabel("%", fontsize = 12)
    plt.title("Time-series Contribution of Portfolio Variance Plot", fontsize = 15)
    plt.legend(df_pct_variance_data.columns)
    plt.show()
    
    ### Graphs for Spectral
    df_rel_comp_sp_data = pd.DataFrame(rel_comp_sp_data[1: ], index=date_vals, columns=data.columns)
    plt.figure(figsize = (12,8), dpi = 120)
    plt.plot(df_rel_comp_sp_data)
    plt.xlabel("Dates", fontsize = 12)
    plt.xticks(rotation = 45)
    plt.ylabel("% Spectral Risk Measure Contribution", fontsize = 12)
    plt.title("Time-series Spectral Risk Measure Contribution Plot", fontsize = 15)
    plt.legend(df_rel_comp_sp_data.columns)
    plt.show()
    
if __name__ == '__main__':
    os.chdir("D:\\Courses\\731\\HWs\\HW3")
    data = pd.read_csv("Five_Stock_Prices.csv").set_index("Date")
    X = load_data(data)
    ### Setting
    alpha = 0.99
    M = 50
    Lambda = 0.94
    theta= 0.96
    gamma = 25
    Vt = 15000000
    N = len(X[0])
    print("Shape: ", X[0].shape)
    ### Parameters
    positions = np.array([[Vt*0.2] * 5] * N)
    
    var_data = np.zeros(shape=(N-M+1, 1))
    mar_var_data = np.zeros(shape=(N-M+1, 5))
    comp_var_data = np.zeros(shape=(N-M+1, 5))
    rel_comp_var_data = np.zeros(shape=(N-M+1, 5))
    
    sp_data = np.zeros(shape=(N-M+1, 1))
    mar_sp_data = np.zeros(shape=(N-M+1, 5))
    comp_sp_data = np.zeros(shape=(N-M+1, 5))
    rel_comp_sp_data = np.zeros(shape=(N-M+1, 5))
    
    pct_variance_data = np.zeros(shape=(N-M+1, 5))
    
    date_vals = []
    
    mu = X[0].iloc[:M].mean()
    cov = X[0].iloc[:M].cov()

    SpectralDensity = quad(f, 0, 1)[0]
    
    ### Main loop
    for i in range(M, N):
        
        ### EWMA estimates ###
        date_vals.append(load_data(data)[1][i+1])
        mu = Lambda * mu + (1 - Lambda) * X[0].iloc[i]
        temp = (np.array(X[0].iloc[i] - mu).reshape(5,1))*(np.array(X[0].iloc[i] - mu).reshape(1,5))
        cov = theta * cov + (1 - theta) * temp
        
        ### risk meansures ###
        ### VaR
        var_data[i+1-M] = -positions[i].dot(mu) + ((positions[i].dot(cov.dot(positions[i])))**0.5)*norm.ppf(alpha)
        ### Marginal VaR
        mar_var_data[i+1-M] = -mu + norm.ppf(alpha)*(np.transpose(cov.dot(positions[i])))/((positions[i].dot(np.transpose(cov.dot(positions[i]))))**0.5)
        ### Components
        comp_var_data[i+1-M] = mar_var_data[i+1-M] * positions[i]
        ### Components Contributions
        rel_comp_var_data[i+1-M] = 100*comp_var_data[i+1-M]/var_data[i+1-M]
        ### Contribution Percentage
        pct_variance_data[i+1-M] = 100*(np.transpose(cov.dot(positions[i]))*positions[i]/(positions[i].dot(np.transpose(cov.dot(positions[i])))))
        
        ### Spectral
        sp_data[i+1-M] = -positions[i].dot(mu) + ((positions[i].dot(cov.dot(positions[i])))**0.5)*SpectralDensity
        ### Marginal Spectral
        mar_sp_data[i+1-M] = -mu + np.array(SpectralDensity)*(np.transpose(cov.dot(positions[i])))/((positions[i].dot(np.transpose(cov.dot(positions[i]))))**0.5)
        ### Components Spectral
        comp_sp_data[i+1-M] = mar_sp_data[i+1-M] * positions[i]
        ### Components Contributions Spectral
        rel_comp_sp_data[i+1-M] = 100 * comp_sp_data[i+1-M]/sp_data[i+1-M]
        
    ###graphs
    graphs()
