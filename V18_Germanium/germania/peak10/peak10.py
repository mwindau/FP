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

y = np.genfromtxt('peak10.txt')

z = np.arange(2755, 2781)

x_plot = np.linspace(2755, 2780, 10000)
plt.xlabel('Kanalnummer')
plt.ylabel('Counts')

plt.plot(z, y, 'r.', label='Messdaten')

params, covariance_matrix = curve_fit(gauss, z, y, p0=[5, 1, 2755])

errors = np.sqrt(np.diag(covariance_matrix))

print(params[0], errors[0])
print(params[1], errors[1])
print(params[2], errors[2])

plt.plot(x_plot, gauss(x_plot, params[0], params[1], params[2]), 'b-', label='Fit des Peaks')

plt.legend()

plt.savefig('peak10.pdf')

#plt.show()
