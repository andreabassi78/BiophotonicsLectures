# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 22:55:35 2021

Assumes an excitation and STED beams as squared cosinusoidal 
and sinusoidal functions with zero (and maximum) in lambda/2NA

Calculates the effective PSF and compares the full width at half maximum
(FWHM) to the thoretical one, as calculated in:

Volker Westphal and Hell, Stefan W. 
"Nanoscale Resolution in the Focal Plane of an Optical Microscope.” 
Physical Review Letters 98, 143903, (2005)

@author: Andrea Bassi
"""

import numpy as np
from matplotlib import pyplot as plt

um = 1
x_extent = 0.26 * um
dx = 0.001 * um
x = np.arange(-x_extent, x_extent, dx)

NA = 1
wavelength = 0.5

delta = np.pi*NA/wavelength*x

Iexc = np.cos(delta)**2 # excitation has the first zero in lambda/2NA
Isted = np.sin(delta)**2 # sted has the first zero in lambda/2NA

ratio = 20 # I_STED_max / I_Sat

PSF = Iexc * np.exp( -ratio * Isted )
PSF_approx = 1 - (ratio+1) * (delta)**2


#%% Find FWHM

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

idx = find_nearest(PSF, np.amax(PSF)*0.5) #find the index corresponging to FWHM

fwhm = 2* np.abs(x[idx])

theoretical_fwhm = 0.45*wavelength/NA/np.sqrt(1+ratio)

print (f'  Measured  FWHM = {fwhm} um')
print (f'Theoretical FWHM = {theoretical_fwhm} um')


#%% Plot graphs
char_size = 12
linewidth = 1

plt.rc('font', family='calibri', size=char_size)

fig = plt.figure(figsize=(5,4), dpi=300)
ax = fig.add_subplot(111)

title = 'plot title'
xlabel = 'x ($\mu$m)'
ylabel = 'intensity'

ax.plot(x, Iexc, 
        linewidth=linewidth,
        linestyle='solid',
        color='green')

ax.plot(x, Isted, 
        linewidth=linewidth,
        linestyle='solid',
        color='red')
 
ax.plot(x, PSF, 
        linewidth=linewidth,
        linestyle='solid',
        color='black')

ax.plot(x, PSF_approx, 
        linewidth=linewidth,
        linestyle='dashed',
        color='black')

ax.set_ylim([0, 1])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)  
ax.set_xlabel(xlabel, size=char_size)
ax.set_ylabel(ylabel, size=char_size)
ax.xaxis.set_tick_params(labelsize=char_size*0.6)
ax.yaxis.set_tick_params(labelsize=char_size*0.6)
 
ax.grid(True, which='major',axis='both',alpha=0.2)   
ax.legend(['exc','sted','effective'],
          loc='center right',frameon = False,
          fontsize=char_size*0.9)

fig.tight_layout()
plt.rcParams.update(plt.rcParamsDefault)
plt.show()