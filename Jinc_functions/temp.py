# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 09:19:23 2021

@author: andrea
"""
from scipy.special import jn, jn_zeros
import matplotlib.pyplot as plt
import numpy as np

a = 1
b = 0.5

def f(x):
    return a**2 * jinc(a*x) - b**2 * jinc(b*x)

def appr_f(x):
    return a**2 * appr_jinc(a*x) - b**2 * appr_jinc(b*x)

def jinc(x):
    # define jinc (r) = 2 * J1 *( pi*r) / (pi*r)
    return 2*jn(1,np.pi*x) / (np.pi*x)

def sinc(x):
    # define jinc (r) = 2 * J1 *( pi*r) / (pi*r)
    return np.sin(np.pi*x) / (np.pi*x)

def appr_jinc(x):
    # for small x values (up to x=0.5) jinc(x) = cos(x/2) 
    # here we use jinc(x) = cos(x/1.22/2) so that the first root is the same      
    return   np.cos(np.pi*x/1.22/2)

def _appr_jinc(x):
    # approximate jinc with a gaussian that goes to e**-2 at x=1
    return np.exp(-2*(x)**2)

def _appr_jinc(x):
    # approximate jinc with a sinc that goes to 0 at 1.22
    return sinc(x/1.22)

X = np.linspace(0.01,2,200)

fig0, ax = plt.subplots()
#ax.plot(X, (a**2)*jinc(a*X))
#ax.plot(X, -(b**2)*jinc(b*X))
ax.plot(X, (a**2-b**2)*sinc(a*X))

ax.plot(X, f(X))
ax.grid(True, which='both')
plt.title('jinc functions difference')


