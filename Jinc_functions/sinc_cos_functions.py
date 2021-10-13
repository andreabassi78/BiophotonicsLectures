# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 09:19:23 2021

@author: andrea
"""
from scipy.special import jn, jn_zeros
import matplotlib.pyplot as plt
import numpy as np

a = 1

def jinc(x):
    # define jinc (r) = 2 * J1 *( pi*r) / (pi*r)
    return 2*jn(1,np.pi*x/a) / (np.pi*x/a)

def sinc(x):
    # define jinc (r) = 2 * J1 *( pi*r) / (pi*r)
    return np.sin(np.pi*x/a) / (np.pi*x/a)

def cos(x):
    # define jinc (r) = 2 * J1 *( pi*r) / (pi*r)
    return np.cos(2*np.pi*x/(2*a))

X = np.linspace(0.01,4,200)

fig0, ax = plt.subplots()
ax.plot(X, sinc(X)**2)
ax.plot(X, cos(X)**2)
ax.grid(True, which='both')
plt.title('compared sinc and cos functions')

