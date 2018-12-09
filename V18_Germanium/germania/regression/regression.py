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

#y = np.genfromtxt('peak1.txt')
#z = np.arange(303, 316)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[111, 1, 303])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#a = ufloat(params[1], errors[1])
#print(a)
#
#y = np.genfromtxt('peak2.txt')
#z = np.arange(610, 619)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[48, 1, 610])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#b = ufloat(params[1], errors[1])
#print(b)
#
#y = np.genfromtxt('peak3.txt')
#z = np.arange(855, 869)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[32, 1, 855])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#c = ufloat(params[1], errors[1])
#print(c)
#
#y = np.genfromtxt('peak4.txt')
#z = np.arange(1022, 1033)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[17, 1, 1022])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#d = ufloat(params[1], errors[1])
#print(d)
#
#y = np.genfromtxt('peak5.txt')
#z = np.arange(1104, 1113)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[16, 1, 1104])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#e = ufloat(params[1], errors[1])
#print(e)
#
#y = np.genfromtxt('peak6.txt')
#z = np.arange(1933, 1947)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[21, 1, 1933])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#f = ufloat(params[1], errors[1])
#print(f)
#
#y = np.genfromtxt('peak7.txt')
#z = np.arange(2152, 2167)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[11, 1, 2152])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#g = ufloat(params[1], errors[1])
#print(g)
#
#y = np.genfromtxt('peak8.txt')
#z = np.arange(2389, 2408)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[4, 1, 2389])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#h = ufloat(params[1], errors[1])
#print(h)
#
#y = np.genfromtxt('peak9.txt')
#z = np.arange(2694, 2711)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[15, 1, 2694])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#i = ufloat(params[1], errors[1])
#print(i)
#
#y = np.genfromtxt('peak10.txt')
#z = np.arange(2755, 2781)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[5, 1, 2755])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#j = ufloat(params[1], errors[1])
#print(j)
#
#y = np.genfromtxt('peak11.txt')
#z = np.arange(3490, 3513)
#params, covariance_matrix = curve_fit(gauss, z, y, p0=[10, 1, 3490])
#errors = np.sqrt(np.diag(covariance_matrix))
#print(params[1], errors[1])
#k = ufloat(params[1], errors[1])
#print(k)
#
#peaks = [a, b, c, d, e, f, g, h, i, j, k]

mittelwerte = [308.89, 613.97, 861.02, 1026.71, 1108.21, 1939.42, 2158.75, 2398.75, 2701.28, 2766.18, 3500.88]
fehler_mittelwerte = [0.02, 0.06, 0.02, 0.17, 0.13, 0.16, 0.20, 0.10, 0.24, 0.11, 0.16]
energien = [121.78, 244.70, 344.30, 411.12, 443.96, 778.90, 867.37, 964.08, 1085.90, 1112.10, 1408.00]

print("Mittelwerte: ", mittelwerte)
print("Fehler Mittelwerte: ", fehler_mittelwerte)

#plt.xlim(760, 780)
#plt.ylim(307, 310)
#plt.plot(mittelwerte, energien, 'r.')


errY = fehler_mittelwerte

plt.errorbar(mittelwerte, energien, yerr=errY, fmt='o', label='Mittelwerte der Gaußpeaks', markersize=3)




#def lineareregression (x, m, n):
 #   return a * m + n

params, covariance_matrix = np.polyfit(mittelwerte, energien, deg=1, cov=True)

errors = np.sqrt(np.diag(covariance_matrix))

print("Steigung der Regressionsgeraden: ", params[0], "Fehler: ", errors[0])
print("Achsenabschnitt der Regressionsgeraden: ", params[1], "Fehler: ", errors[1])

x_plot = np.linspace(0, 4000, 10000)

plt.plot(x_plot, params[0] * x_plot + params[1], label='Fit', linewidth=1)
plt.xlabel('Kanalnummer')
plt.ylabel('Counts')

plt.legend()

plt.savefig('regression.pdf')

































#lol
