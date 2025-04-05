# Power-Spectral-Analysis
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('interpolated_data2(ULL_1995_B).txt')


Bx = data[:, 1]
By = data[:, 2]
Bz = data[:, 3]


n = len(Bx)

t = np.arange(n)

Xx = np.fft.fft(Bx)
Xy = np.fft.fft(By)
Xz = np.fft.fft(Bz)


PBx = np.abs(Xx)**2 / n
PBy = np.abs(Xy)**2 / n
PBz = np.abs(Xz)**2 / n


PBx = PBx[:n//2]
PBy = PBy[:n//2]
PBz = PBz[:n//2]

frequencies = np.fft.fftfreq(n, 1/n)[:n//2]
Pfactor = ((1/n)*(frequencies))[:n//2]

PB_AVG = (PBx + PBy + PBz) / 3

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

window_size = 50
sma_PBx = moving_average(PBx, window_size)
sma_frequencies = moving_average(Pfactor, window_size)  # Match the frequency array for plotting

restricted_indices = (sma_frequencies >= 0.001) & (sma_frequencies <= 0.01)
restricted_sma_frequencies = sma_frequencies[restricted_indices]
restricted_sma_PBx = sma_PBx[restricted_indices]

poly_order = 1  
coeffs = np.polyfit(np.log(restricted_sma_frequencies), np.log(restricted_sma_PBx), poly_order)
poly = np.poly1d(coeffs)

slope = coeffs[0]

x_fit_restricted = np.linspace(min(restricted_sma_frequencies), max(restricted_sma_frequencies), 1000)
y_fit_restricted = np.exp(poly(np.log(x_fit_restricted)))

x_fit_whole = np.linspace(min(sma_frequencies), max(sma_frequencies), 1000)
y_fit_whole = np.exp(poly(np.log(x_fit_whole)))

print("Slope (Power-law exponent) for restricted domain:", slope)

plt.figure(figsize=(12, 6))

plt.loglog(sma_frequencies, sma_PBx, label='Moving Average Bx', color='Pink')
plt.loglog(x_fit_whole, y_fit_whole, label='Polynomial Fit (Whole Domain)', color='Green', linestyle='--')


plt.loglog(restricted_sma_frequencies, restricted_sma_PBx, 'o', label='Restricted Domain Data', color='Red')
plt.loglog(x_fit_restricted, y_fit_restricted, label=f'Polynomial Fit (Restricted Domain, Slope: {slope:.2f})', color='Blue')

plt.text(0.002, max(sma_PBx), f'Slope: {slope:.2f}', fontsize=12, color='blue')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (B_AVG)')
plt.title('Power Spectrum and Polynomial Fit')
plt.show()
