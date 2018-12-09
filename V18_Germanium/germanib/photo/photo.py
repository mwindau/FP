import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def gauss (x, amplitude, mu, sigma):
    return amplitude * np.exp(-1/2 * (x - mu)**2/sigma**2)

a = np.arange(1637, 1660)

b = np.genfromtxt('photo.txt')

c = a * 0.40298 - 2.654



d = 1637 * 0.40298 - 2.654

x_plot = np.linspace(656, 666, 10000)



plt.plot(c, b, 'r.', label='Messdaten')
plt.xlabel('Energie in keV')
plt.ylabel('Counts')

params, covariance_matrix = curve_fit(gauss, c, b, p0=[2552, 661.46, 0.9])

errors = np.sqrt(np.diag(covariance_matrix))

print(params[0], errors[0])
print(params[1], errors[1])
print(params[2], errors[2])

#plt.plot(x_plot, gauss(x_plot, 2552, 661.55, 0.9))
plt.plot(x_plot, gauss(x_plot, params[0], params[1], params[2]), 'b-', label='Fit des Peaks')
plt.legend()

plt.savefig('photoenergien.pdf')
plt.clf()

sigma = ufloat(params[2], errors[2])

hwbberechnet = 2 * unp.sqrt(2 * unp.log(2)) * sigma
zwbberechnet = 1.823 * hwbberechnet
vahaltnis = zwbberechnet/ hwbberechnet

print('Halbwertsbreite berechnet: ', hwbberechnet)
print('Zehntelwertsbreite berechnet: ', zwbberechnet)
print('Verhältnis: ', vahaltnis)

def ehwb(E_gamma):
    return 2.35 * unp.sqrt(0.1 * E_gamma * 0.0029)

gaussmittelwert = ufloat(661.52, 0.01)


print('Halbwertsbreite aus Energieformel: ', ehwb(gaussmittelwert))
