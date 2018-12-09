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
b = np.genfromtxt('le.txt')
#plt.plot(a, b)
#plt.savefig('lekanal.pdf')
#plt.clf()


e = a * measy + neasy
plt.plot(e, b)
plt.xlabel('Energie in keV')
plt.ylabel('Counts')
plt.savefig('leenergie.pdf')
plt.clf()

e = a * measy + neasy
plt.yscale('log')
plt.plot(e, b)
plt.xlabel('Energie in keV')
plt.ylabel('Counts')
plt.savefig('leenergielog.pdf')
plt.clf()

#e = a * measy + neasy
#plt.xlim(0, 1000)
#plt.ylim(0, 1500)
#plt.plot(e, b)
#plt.savefig('leenergiezoomlinks.pdf')
#plt.clf()
#
#e = a * measy + neasy
#plt.xlim(1000, 2000)
#plt.ylim(0, 700)
#plt.plot(e, b)
#plt.savefig('leenergiezoomrechts.pdf')
#plt.clf()
#
#plt.xlim(0, 1300)
#plt.plot(a, b)
#plt.savefig('lekanalzoomlinks.pdf')
#plt.clf()
#
#plt.xlim(100, 300)
#plt.plot(a, b)
#plt.savefig('lekanalzoomkrasslinks.pdf')
#plt.clf()
#
#plt.xlim(1500, 4000)
#plt.ylim(0, 500)
#plt.plot(a, b)
#plt.savefig('lekanalzoommitte.pdf')
#plt.clf()
#
#plt.xlim(3090, 7000)
#plt.ylim(0, 400)
#plt.plot(a, b)
#plt.savefig('lekanalzoomrechts.pdf')
#plt.clf()
#
