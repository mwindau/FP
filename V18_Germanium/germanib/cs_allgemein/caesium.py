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

a = np.arange(1, 8192)

b = np.genfromtxt('caesium.txt', unpack=True)

c = a * 0.40298 - 2.654


##mit kanal
#plt.plot(a, b)
#plt.savefig('caesium.pdf')
#plt.clf()

#log mit kanal
#plt.yscale('log')
#plt.plot(a, b)
#plt.savefig('caesiumlog.pdf')
#plt.clf()

#mit enrgie
plt.plot(c, b)
plt.xlabel('Energie in keV')
plt.ylabel('Counts')
plt.savefig('casenergie.pdf')
plt.clf()

#log mit energie
plt.yscale('log')
plt.plot(c, b)
plt.xlabel('Energie in keV')
plt.ylabel('Counts')
plt.savefig('casenergielog.pdf')
plt.clf()

#log mit energie und schnitt
plt.yscale('log')
plt.xlim(0, 750)
plt.plot(c, b)
plt.savefig('casenlogschnitt.pdf')
plt.clf()
