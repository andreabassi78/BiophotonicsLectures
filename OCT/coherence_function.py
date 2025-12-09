import numpy as np
from numpy import pi, log, sqrt
import matplotlib.pyplot as plt

um = 1.0  # micrometer unit
lambda0 = 1*um     # central wavelength
n = 1.33 # refractive index at the sample (and at the reference mirror)
k0 = n / lambda0 # wavenumber
Lc = 2.0*um  # coherence length

# z-axis for optical path difference: covers some coherence lengths on each side
z_min = -32.0*um
z_max =  32.0*um
N = 4095  # number of points
z = np.linspace(z_min, z_max, N)

# Coherence function defined as a Gaussian envelope with FWHM = Lc
sigma = Lc / (2*sqrt(log(2)))
sigma = Lc / 2
Gamma = np.exp(-(z/sigma)**2) * np.cos(2*np.pi*k0*z)

# Power spectrum S(k) via FFT
dz = z[1] - z[0]
S_k = np.fft.fft(np.fft.ifftshift(Gamma))
S_k = np.fft.fftshift(S_k)

# Spatial frequency axis k (cycles/µm)
k_axis = np.fft.fftshift(np.fft.fftfreq(N, d=dz))
P_k = np.abs(S_k)

# Bandwidth calculations
Delta_k = 4*log(2) / (pi* Lc)   # Δk obtained as the FWHM of the Gaussian function (Fourier transform of the coherence function)
sigma_k = 2 / (pi*Lc)                 # Δk obtained as the width at 1/e of the Gaussian function (Fourier transform of the coherence function)   
sigma_lambda = sigma_k * lambda0**2 / n  # Δλ

# Convert S(k) to wavelength-domain spectrum S(λ)
mask = k_axis > 0
k_pos = k_axis[mask]
lambda_axis = n / k_pos        # λ = n/k
P_lambda = P_k[mask]

# Sort by wavelength
order = np.argsort(lambda_axis)
lambda_axis = lambda_axis[order]
P_lambda = P_lambda[order]

plt.figure()
plt.plot(z, Gamma)
plt.title(f"Coherence function Γ(z)\nLc = {Lc:.2f} µm (2σ)")
plt.xlabel("z (µm)")
plt.ylabel("Amplitude (a.u.)")
plt.axvline(-Lc/2, color='k', linestyle=':', linewidth=1)
plt.axvline( Lc/2, color='k', linestyle=':', linewidth=1)
plt.grid(True, linestyle=":", linewidth=0.5)

plt.figure()
plt.plot(k_axis, P_k)
plt.title("Power spectrum S(k), bandwidth Δk = {:.3f} 1/µm".format(2*sigma_k))
plt.xlabel("k (1/µm)")
plt.ylabel("S(k) (a.u.)")
plt.axvline(k0 - sigma_k, color='k', linestyle=':', linewidth=1)
plt.axvline(k0 + sigma_k, color='k', linestyle=':', linewidth=1)
plt.grid(True, linestyle=":", linewidth=0.5)
plt.xlim(0, 2*k0)

plt.figure()
plt.plot(lambda_axis, P_lambda)
plt.title("Power spectrum S(λ), bandwidth Δλ = {:.1f} nm".format(2*sigma_lambda*1e3))
plt.xlabel("λ (µm)")
plt.ylabel("S(λ) (a.u.)")
plt.grid(True, linestyle=":", linewidth=0.5)
plt.axvline(lambda0 - sigma_lambda, color='k', linestyle=':', linewidth=1)
plt.axvline(lambda0 + sigma_lambda, color='k', linestyle=':', linewidth=1)
plt.xlim(0, 2*lambda0)

plt.show()
