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

b = np.genfromtxt('rueck.txt')

a = np.arange(460, 506)

plt.plot(a, b, 'r.', markersize=3, label='Messdaten')


params, covariance_matrix = curve_fit(gauss, a, b, p0=[127, 480, 0.2])

errors = np.sqrt(np.diag(covariance_matrix))

print(params[0], errors[0])
print(params[1], errors[1])
print(params[2], errors[2])


x_plot = np.linspace(460, 505, 10000)

#plt.plot(x_plot, gauss(x_plot, params[0], params[1], params[2]), 'b-', label='Fit des Peaks')
plt.legend()

plt.savefig('rueckkanal.pdf')
plt.clf()

m = ufloat(0.40298, 0.00002)
n = ufloat(-2.65400, 0.04789)

measy = 0.40298
neasy = -2.654

c = a * measy + neasy

plt.plot(c, b, 'b.')
plt.xlabel('Energie in keV')
plt.ylabel('Counts')
plt.savefig('rueckenergie.pdf')

lagepeak = 471 * m + n
print('Lage des Rückstreupeaks (abgelesen): ', lagepeak)

e_gamma = ufloat(661.52, 0.01)
m_e = ufloat(const.physical_constants["electron mass energy equivalent in MeV"][0], const.physical_constants["electron mass energy equivalent in MeV"][2])
elektronenmasse = m_e * 1000

def streuenergie(energie):
    return energie/(1 + 2 * (energie/elektronenmasse))

lagepeakberechnet = streuenergie(e_gamma)




print('Lage des Rückstreupeaks (berechnet): ', lagepeakberechnet)
