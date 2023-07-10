# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:54:57 2023

@author: aboag
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK,space_eval

#%%
"""
Suppose we want to model the spread of a contagious disease in a population. 
One commonly used mechanistic model is the SIR model, which divides the population into three groups:

Susceptible (S): those who can catch the disease.
Infected (I): those who have caught the disease and can spread it.
Recovered (R): those who have recovered from the disease and are now immune.

dS/dt = -beta * S * I
dI/dt = beta * S* I - gamma * I
dR/dt = gamma * I

where: beta is the transmission rate of the disease
       gamma is the recovery rate
"""

#%% Define functions/models 
#Define the SIR model
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

#Define a function to fit the data
def fit_odeint(t, beta, gamma):
    return odeint(sir_model, (S0, I0, R0), t, args=(beta, gamma))[:,1]

#Sample data and initial conditions
t_data = np.linspace(0, 14, 15) # Days of observation
I_data = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4, 2]) # Infected individuals

N = 763             # Total population
I0 = 3              # Initial number of infected individuals
R0 = 0              # Initial number of recovered individuals
S0 = N - I0 - R0    # Initial number of susceptible individuals

#%% Hyperopt for finding optimal parameters
params = {
    'beta_param': hp.uniform('beta_param', 0.001,1),
    'gamma_param': hp.uniform('gamma_param', 0.001,1),
    }

# Loss function
def loss(params):
    beta = params['beta_param']
    gamma = params['gamma_param']
    fitted = fit_odeint(t_data, beta, gamma)
    mse = mean_squared_error(I_data, fitted)
    return {'loss': mse, 'status': STATUS_OK}

trials = Trials()
Best = fmin(fn=loss, 
            space=params, 
            algo=tpe.suggest, 
            max_evals=10000,
            trials=trials)
print('Best hyperparameters: ', space_eval(params,Best))

#Hyperopt best parameters
#'beta_param': 0.002238741669659896, 'gamma_param': 0.4531449898895715
#274.5612634215566

#%% Redefine model and solve with optimal parameters
time_data = np.linspace(0,14,100) 

# Define the SIR model
def sir_model_fitted(y, t, beta=0.002238741669659896, gamma=0.4531449898895715):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

solution = odeint(sir_model_fitted, (S0, I0, R0), time_data)

#%% Plot results
plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.7)
plt.plot(t_data, I_data, 'o', label='original')
plt.plot(time_data, solution[:,1], label='fitted')
plt.title("Fit of SIR model to data")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.plot(time_data, solution, label=['susceptible','infected','recovered'])
plt.title("SIR model")
plt.ylabel("Population")
plt.xlabel("Days")
plt.legend(loc='best')
plt.show()

#%%
"""
Let's consider a reaction scheme: a consecutive first-order reaction 
where species A transforms into B, which further transforms into C.

The reaction scheme is: A --> B --> C

Fit the model for the rate constants k1 and k2
"""
#%% Define the models
# function for reaction model
def reaction_model(y, t, k1, k2):
    A, B, C = y
    dAdt = -k1 * A
    dBdt = k1 * A - k2 * B
    dCdt = k2 * B
    return [dAdt, dBdt, dCdt]

# fucntion to fit the data
def fit_odeint_(t, k1, k2):
    return odeint(reaction_model, (A0, B0, C0), t, args=(k1, k2))[:,1]

# sample data and initial conditions
tm_data = np.linspace(0,10,100)
A0 = 10
B0 = 0
C0 = 0 
k1_true = 0.5
k2_true = 0.1

ABC_data = odeint(reaction_model, (A0, B0, C0), tm_data, args=(k1_true, k2_true)) # Generate true data
ABC_data += np.random.normal(0, 0.1, ABC_data.shape) # Add some noise

#%% Hyperopt for finding optimal parameters
params_rxn = {
    'k1_param': hp.uniform('k1_param', 0.001,1),
    'k2_param': hp.uniform('k2_param', 0.001,1),
    }

# Loss function
def loss_rxn(params):
    k1 = params['k1_param']
    k2 = params['k2_param']
    fitted = fit_odeint_(tm_data[:69], k1, k2)
    mse = mean_squared_error(ABC_data[:69,1], fitted)
    return {'loss': mse, 'status': STATUS_OK}

trials = Trials()
Best = fmin(fn=loss_rxn, 
            space=params_rxn, 
            algo=tpe.suggest, 
            max_evals=10000,
            trials=trials)
print('Best hyperparameters: ', space_eval(params,Best))

#Hyperopt best parameters (using all the data for fitting the model)
#{'k1_param': 0.5020527339701352, 'k2_param': 0.09966421672345421}
#0.009912599292233258

#Hyperopt best parameters (using 70% of the data)
#{'k1_param': 0.5047365839880686, 'k2_param': 0.10015746332576542}
#0.011129382057221898

#%% Use optimal values to fit model
k1_fitted = 0.5020527339701352
k2_fitted = 0.09966421672345421
ABC_fitted = odeint(reaction_model, (A0, B0, C0), tm_data, args=(k1_fitted, k2_fitted))

#%% Plotting
plt.figure(figsize=(14,10), dpi=1000)
plt.plot(tm_data, ABC_data, 'o', label=['A_true','B_true','C_true'])
plt.plot(tm_data, ABC_fitted, label=['A_fit','B_fit','C_fit'])
plt.title("Concentration Profiles")
plt.ylabel("Concentration")
plt.xlabel("time")
plt.legend(loc='best')
plt.show()
#%%
k1_fitted_70 = 0.5047365839880686
k2_fitted_70 = 0.10015746332576542
ABC_fitted_70 = odeint(reaction_model, (A0, B0, C0), tm_data, args=(k1_fitted_70, k2_fitted_70))
#%%
plt.figure(figsize=(14,10), dpi=1000)
plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.25)
plt.plot(tm_data[:70], ABC_data[:70], 'o', label=['A_true','B_true','C_true'])
plt.plot(tm_data[:70], ABC_fitted_70[:70], label=['A_fit_70','B_fit_70','C_fit_70'])
plt.title("Concentration Profiles (using 70% of data)")
plt.ylabel("Concentration")
plt.xlabel("time")
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.plot(tm_data[70:], ABC_data[70:], 'o', label=['A_true','B_true','C_true'])
plt.plot(tm_data[70:], ABC_fitted_70[70:], label=['A_fit_30','B_fit_30','C_fit_30'])
plt.title("Concentration Profiles (using 30% for validation)")
plt.ylabel("Concentration")
plt.xlabel("time")
plt.legend(loc='best')
plt.show()

