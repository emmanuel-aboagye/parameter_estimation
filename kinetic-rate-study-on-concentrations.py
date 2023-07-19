# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:53:34 2023

@author: aboag
"""

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#%% Define model and solve ODE
"""
Solve the reaction equation A ----> B <========> C
Perform sensitivity analysis to evaluate how each rate constant affect each 
concentration. 
ka = 1, kb = 0.5, kc = 0.25, Ca(0) = 3, Cb(0) = 1, Cc(0) = 0
"""

ka = 1 
kb = 0.5
kc = 0.25

def kinetic_model(X, t, ka, kb, kc):
    
    # define state variables
    Ca = X[0]
    Cb = X[1]
    Cc = X[2]
    
    # define differential equations
    dCadt = -ka * Ca
    dCbdt = ka * Ca - kb * Cb + kc * Cc
    dCcdt = kb * Cb - kc * Cc
    
    return [dCadt, dCbdt, dCcdt]


# initial conditions and time internval
time = np.linspace(0,20,100)
initial_conditions = [3, 1, 0]

# solve odes 
sol = odeint(kinetic_model, initial_conditions, time, args=(ka, kb, kc))

# plot resutls 

#plt.subplot(2,1,1)
#plt.subplots_adjust(hspace=0.7)
plt.figure(figsize=(7,5), dpi=700)
plt.plot(time, sol, label=['Ca', 'Cb', 'Cc'])
plt.ylabel("Concentration")
plt.xlabel("time")
plt.legend()
plt.show()
#%% Perform sensitivity analysis on how each rate parameter affects each concentration species
"""
dCi/dkj = P(Ci, kj), where i = a,b,c and j = a,b,c. Thus we will have to solve 9 differential equations to 
see how each rate constant has an impact on the individual concentrations
"""
def sensitivity_model(X, t):
    
    # define state variables
    Caka = X[0]
    Cakb = X[1]
    Cakc = X[2]
    Cbka = X[3]
    Cbkb = X[4]
    Cbkc = X[5]
    Ccka = X[6]
    Cckb = X[7]
    Cckc = X[8]
    Ca   = X[9]
    Cb   = X[10]
    Cc   = X[11]
    
    # define differential equations
    dP_Cakadt = - Ca - ka * Caka
    dP_Cakbdt = - ka * Cakb
    dP_Cakcdt = - ka * Cakc
    dP_Cbkadt = Ca + ka * Caka - kb * Cbka + kc * Ccka
    dP_Cbkbdt = ka * Cakb - Cb - kb * Cbkb + kc * Cckb
    dP_Cbkcdt = ka * Cakc - kb * Cbkc + Cc + kc * Cckc
    dP_Cckadt = kb * Cbka - kc * Ccka
    dP_Cckbdt = Cb + kb * Cbkb - kc * Cckb
    dP_Cckcdt = kb * Cbkc - Cc - kc * Cckc
    dCadt = -ka * Ca
    dCbdt = ka * Ca - kb * Cb + kc * Cc
    dCcdt = kb * Cb - kc * Cc
    
    return [dP_Cakadt, dP_Cakbdt, dP_Cakcdt, dP_Cbkadt, dP_Cbkbdt, dP_Cbkcdt, dP_Cckadt, dP_Cckbdt, dP_Cckcdt, dCadt, dCbdt, dCcdt]
    
# initial conditions
initial_conditions_sen = [0,0,0,0,0,0,0,0,0,3,1,0]

# solve odes 
sol_sen = odeint(sensitivity_model, initial_conditions_sen, time)

#plotting
plt.figure(figsize=(16,12), dpi=700)
plt.subplots_adjust(hspace=0.5)
plt.subplot(3,3,1)
plt.plot(time, sol_sen[:,0])
plt.ylabel("Sensitivity of Ca wrt ka")
plt.xlabel("time")

plt.subplot(3,3,2)
plt.plot(time, sol_sen[:,1])
plt.ylabel("Sensitivity of Ca wrt kb")
plt.xlabel("time")

plt.subplot(3,3,3)
plt.plot(time, sol_sen[:,2])
plt.ylabel("Sensitivity of Ca wrt kc")
plt.xlabel("time")

plt.subplot(3,3,4)
plt.plot(time, sol_sen[:,3])
plt.ylabel("Sensitivity of Cb wrt ka")
plt.xlabel("time")

plt.subplot(3,3,5)
plt.plot(time, sol_sen[:,4])
plt.ylabel("Sensitivity of Cb wrt kb")
plt.xlabel("time")

plt.subplot(3,3,6)
plt.plot(time, sol_sen[:,5])
plt.ylabel("Sensitivity of Cb wrt kc")
plt.xlabel("time")

plt.subplot(3,3,7)
plt.plot(time, sol_sen[:,6])
plt.ylabel("Sensitivity of Cc wrt ka")
plt.xlabel("time")

plt.subplot(3,3,8)
plt.plot(time, sol_sen[:,7])
plt.ylabel("Sensitivity of Cc wrt kb")
plt.xlabel("time")

plt.subplot(3,3,9)
plt.plot(time, sol_sen[:,8])
plt.ylabel("Sensitivity of Cc wrt kc")
plt.xlabel("time")
plt.show()
#%% Introduce some pertubation for each kinetic parameter and resimulate the model
"""
we will use a range of +10% to -10% for each parameter
"""

fraction = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10]
Ca_results = [None] * len(fraction)
Cb_results = [None] * len(fraction)
Cc_results = [None] * len(fraction)
# solutions = [None] * len(fraction)

for i in range(len(fraction)):
    print('running iteration', fraction[i] * 100, '%')
    ka_ = ka * (1 + fraction[i])
    kb_ = kb * (1 + fraction[i])
    kc_ = kc * (1 + fraction[i])
    sol_resim = odeint(kinetic_model, initial_conditions, time, args=(ka_, kb_, kc_))
    # solutions[i] = sol_resim
    Ca_results[i] = sol_resim[:,0]
    Cb_results[i] = sol_resim[:,1]
    Cc_results[i] = sol_resim[:,2]

#plotting
labels_ = [str(100*i)+'%' for i in fraction]

plt.figure(figsize=(16,12), dpi=700)
plt.subplots_adjust(wspace=0.5)
plt.subplot(3,1,1)
for j in range(len(Ca_results)):
    plt.plot(time, Ca_results[j])
plt.ylabel("Concentration of A")
plt.xlabel("time")
plt.legend(labels_)

plt.subplot(3,1,2)
for k in range(len(Cb_results)):
    plt.plot(time, Cb_results[k])
plt.ylabel("Concentration of B")
plt.xlabel("time")
plt.legend(labels_)
plt.subplot(3,1,3)
for l in range(len(Cc_results)):
    plt.plot(time, Cc_results[l])
plt.ylabel("Concentration of C")
plt.xlabel("time")
plt.legend(labels_)
plt.show()
#%% 
