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


e_gamma = ufloat(661.52, 0.01)
m_e = ufloat(const.physical_constants["electron mass energy equivalent in MeV"][0], const.physical_constants["electron mass energy equivalent in MeV"][2])
elektronenmasse = m_e * 1000

e_gammag = 661.52
m_eg = 511

epsilon = e_gammag/m_eg
thomson = 6.652 * 10**(-29)


def spektrum(E, amplitude):
    return amplitude * thomson * 1/(m_eg *epsilon**2) * (2 + (E / (e_gammag - E))**2 * ((1/epsilon**2) + (e_gammag - E)/e_gammag - 2/epsilon * (e_gammag -E)/e_gammag))

b = np.genfromtxt('contifit.txt')
a = np.arange(1100, 1181)
a_energie = a * 0.40298 - 2.654

plt.plot(a_energie, b, 'r.')


plt.xlabel('Energie in keV')
plt.ylabel('Counts')


params, covariance_matrix = curve_fit(spektrum, a, b, p0=[10**32.3])
#
errors = np.sqrt(np.diag(covariance_matrix))
#
print(params[0], errors[0])

x_plot = np.linspace(1100, 1180, 10000)
x_plot_ene = x_plot * 0.40298 - 2.654
#
#plt.plot(x_plot_ene, spektrum(x_plot, params[0]), 'b-', label='Fit des Peaks')

plt.plot(x_plot_ene, spektrum(x_plot_ene, 10**32.35))

#print(spektrum(460, 10**32.4))


plt.savefig('contifitrichtig.pdf')
































#platz
