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

m = ufloat(0.40298, 0.00002)
n = ufloat(-2.65400, 0.04789)

measy = 0.40298
neasy = -2.654

a = np.arange(1109, 1190)
b = np.genfromtxt('contifit.txt')

a_neu = a * measy + neasy



plt.plot(a_neu, b)
plt.xlabel('Energie in keV')
plt.ylabel('Counts')

e_gammag = 661.52
m_eg = 511

epsilon = e_gammag/m_eg
thomson = 6.652 * 10**(-29)

def spektrum(E, amplitude):
    return amplitude * thomson * 1/(m_eg *epsilon**2) * (2 + (E / (e_gammag - E))**2 * ((1/epsilon**2) + (e_gammag - E)/e_gammag - 2/epsilon * (e_gammag -E)/e_gammag))

params, covariance_matrix = curve_fit(spektrum, a, b, p0=[10**32.35])
#
errors = np.sqrt(np.diag(covariance_matrix))
#
print(params[0], errors[0])

x_plot = np.linspace(443, 478, 10000)

plt.plot(x_plot, spektrum(x_plot, params[0]))

plt.savefig('comptonecke.pdf')





























#platz
