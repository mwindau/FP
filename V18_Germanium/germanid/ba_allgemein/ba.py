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

a = np.arange(1, 8192)
b = np.genfromtxt('ba.txt')
#plt.plot(a, b)
#plt.savefig('bakanal.pdf')
#plt.clf()
#
#a = np.arange(1, 1001)
#b = np.genfromtxt('bazoom.txt')
#plt.plot(a, b)
#plt.savefig('bakanalzoom.pdf')
#plt.clf()
#
#a = np.arange(1, 8192)
#b = np.genfromtxt('ba.txt')
#plt.yscale('log')
#plt.plot(a, b)
#plt.savefig('bakanallog.pdf')
#plt.clf()


e = a * measy + neasy
plt.plot(e, b)
plt.xlabel('Energie in keV')
plt.ylabel('Counts')
plt.savefig('baenergie.pdf')
plt.clf()

e = a * measy + neasy
plt.yscale('log')
plt.plot(e, b)
plt.xlabel('Energie in keV')
plt.ylabel('Counts')
plt.savefig('baenergielog.pdf')
plt.clf()

#a = np.arange(1, 1001)
#b = np.genfromtxt('bazoom.txt')
#e = a * measy + neasy
#plt.plot(e, b)
#plt.savefig('baenergiezoom.pdf')
#plt.clf()
