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

a = np.arange(1100, 1301)
b = np.genfromtxt('kante.txt')
#plt.plot(a, b)
#plt.savefig('kantekanal.pdf')
#plt.clf()
#
#
#f = np.arange(1175, 1201)
#g = np.genfromtxt('close.txt')
#plt.plot(f, g)
#plt.savefig('kantekanalclose.pdf')
#plt.clf()
#
#
#e = a * measy + neasy
#plt.plot(e, b)
#plt.xlabel('Energie in keV')
#plt.ylabel('Counts')
#plt.savefig('kanteenergie.pdf')
#plt.clf()

#lagekante = 1189 * m + n
#print('Lage der Comptonkante (abgelesen): ', lagekante)


e_gamma = ufloat(661.52, 0.01)
m_e = ufloat(const.physical_constants["electron mass energy equivalent in MeV"][0], const.physical_constants["electron mass energy equivalent in MeV"][2])
elektronenmasse = m_e * 1000

def kante(energie):
    return energie * (2 * energie / elektronenmasse)/(1 + 2 * energie / elektronenmasse)





#lagekanteberechnet = kante(e_gamma)
#print('Lage der Comptonkante (berechnet): ', lagekanteberechnet)







c = np.arange(1, 1401)
j = c * 0.40298 - 2.654
d = np.genfromtxt('kontinuum.txt')
plt.plot(j, d, 'r.')






e_gammag = 661.52
m_eg = 511

epsilon = e_gammag/m_eg
thomson = 6.652 * 10**(-29)

def spektrum(E, amplitude):
    return amplitude * thomson * 1/(m_eg *epsilon**2) * (2 + (E / (e_gammag - E))**2 * ((1/epsilon**2) + (e_gammag - E)/e_gammag - 2/epsilon * (e_gammag -E)/e_gammag))


a_eng =  noms(m) * a + noms(n)
print(  a_eng[a_eng<=480], b[a_eng<=480])
parms, cov = curve_fit(spektrum, a_eng[a_eng<=480], b[a_eng<=480], p0=[10**32.5])

print(parms, cov)


x_plot = np.arange(20, 470)
#x_plot_ene = 0.40298  * x_plot - 2.654

#z = spektrum(x_plot)
#
#plt.plot(x_plot, z)
#
#print(spektrum(100))

x_plot_lin = np.linspace(20, 470, 10000)

plt.plot(x_plot_lin, spektrum(x_plot_lin, 10**32.5))



plt.savefig('kontinuum.pdf')
plt.clf()


b = np.genfromtxt('contifit.txt')
a = np.arange(1100, 1181)
a_energie = a * 0.40298 - 2.654

plt.plot(a_energie, b, 'r.')


params, covariance_matrix = curve_fit(spektrum, a, b, p0=[80])
#
errors = np.sqrt(np.diag(covariance_matrix))
#
print(params[0], errors[0])

x_plot = np.linspace(1100, 1180, 10000)
x_plot_ene = x_plot * 0.40298 - 2.654

plt.plot(x_plot_ene, spektrum(x_plot, params[0]), 'b-', label='Fit des Peaks')
#plt.legend()

plt.savefig('contifit.pdf')
