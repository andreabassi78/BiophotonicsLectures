import numpy as np
import matplotlib.pyplot as plt



um = 1.0  # micrometer unit
lambda0 = 1*um     # central wavelength
Delta_lambda = 0.2*um # Bandwidth of the source
n = 1.33 # refractive index at the sample (and at the reference mirror)
k0 = n / lambda0 # wavenumber

# Calculate the coherence length from the wavelength bandwidth
Lc = lambda0**2 / (n * Delta_lambda)

# z-axis for optical path difference: covers some coherence lengths on each side
z_min = -32.0*um
z_max =  32.0*um
N = 4095  # number of points
z = np.linspace(z_min, z_max, N)

# Coherence function defined as a Gaussian envelope with waist = Lc/2
sigma = Lc / 2
Gamma = np.exp(-(z/sigma)**2) * np.cos(2*np.pi*k0*z)

# defined a sample reflecity r(z) as 3 delta funtions in depth
r = np.zeros_like(z)
depths = [-5*um, 10*um, 15*um]   
amplitudes = [1.0, 0.5, 0.8]
for depth, amp in zip(depths, amplitudes):
    idx = (np.abs(z - depth)).argmin()
    r[idx] = amp            


# calculater the interference signal as the convolution of r(z) and Gamma(z)
I_SR = 1 + np.convolve(r, Gamma, mode='same') * (z[1] - z[0])


# Power spectrum S(k) via FFT
dz = z[1] - z[0]
S_k = np.fft.fft(np.fft.ifftshift(I_SR))
S_k = np.fft.fftshift(S_k)

# Spatial frequency axis k (cycles/µm)
k_axis = np.fft.fftshift(np.fft.fftfreq(N, d=dz))
P_k = np.real(S_k)

Delta_k = 1 / Lc  # predicted Δk

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
plt.plot(z, I_SR)
plt.title(f"Coherence function Γ(z)\nLc = {Lc:.2f} µm (2σ)")
plt.xlabel("z (µm)")
plt.ylabel("Amplitude (a.u.)")
plt.axvline(-sigma, color='k', linestyle=':', linewidth=1)
plt.axvline( sigma, color='k', linestyle=':', linewidth=1)
plt.grid(True, linestyle=":", linewidth=0.5)

plt.figure()
plt.plot(k_axis, P_k)
plt.title("Power spectrum S(k)")
plt.xlabel("k (1/µm)")
plt.ylabel("S(k) (a.u.)")
plt.axvline(k0 - Delta_k/2, color='k', linestyle=':', linewidth=1)
plt.axvline(k0 + Delta_k/2, color='k', linestyle=':', linewidth=1)
plt.grid(True, linestyle=":", linewidth=0.5)
plt.xlim(0.5*k0, 1.5*k0)

plt.figure()
plt.plot(lambda_axis, P_lambda)
plt.title("Power spectrum S(λ)")
plt.xlabel("λ (µm)")
plt.ylabel("S(λ) (a.u.)")
plt.grid(True, linestyle=":", linewidth=0.5)
plt.axvline(lambda0 - Delta_lambda/2, color='k', linestyle=':', linewidth=1)
plt.axvline(lambda0 + Delta_lambda/2, color='k', linestyle=':', linewidth=1)
plt.xlim(lambda0 - 0.5*lambda0, lambda0 + 0.5*lambda0)

plt.show()
