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

#zaehlraten = [14012, 2661, 5726, 514, 597, 1185, 369, 676, 596, 787, 809]
b = ufloat(726.64, 24.86)
c = ufloat(1446.54, 14.38)
d = ufloat(97.10, 6.70)
e = ufloat(115.79, 6.33)
f = ufloat(191.11, 9.88)
g = ufloat(55.13, 2.94)
h = ufloat(144.52, 4.03)
i = ufloat(82.52, 4.45)
j = ufloat(110.40, 2.92)
k = ufloat(108.80, 3.83)
gausszaehlraten = [b, c, d, e, f, g, h, i, j, k]
wahrscheinlichkeiten = [0.076, 0.265, 0.022, 0.031, 0.129, 0.042, 0.146, 0.102, 0.136, 0.210]
aktivitaet = ufloat(1633, 24)

def activity(zrate, aktiv, wkeit):
    return zrate * (1/0.022) * (1/aktiv) * (1/wkeit) * (1/3960)

a = ufloat(4650.32, 68.32)

print("Aktivitaet erster Wert: ", activity(a, aktivitaet, 0.286))

#for i in range(11):
 #   q = activity(zaehlraten[i], aktivitaet, wahrscheinlichkeiten[i])
    #np.append(effizienz, q)
  #  print(q)

for i in range(10):
    r = activity(gausszaehlraten[i], aktivitaet, wahrscheinlichkeiten[i])
    print(r)

#energien = [309, 614, 861, 1027, 1108, 1939, 2159, 2398, 2701, 2766, 3501]
gaussenergien = [613.97, 861.02, 1026.71, 1108.21, 1939.42, 2158.75, 2398.75, 2701.28, 2766.18, 3500.88]
#steigung = ufloat(0.40298, 0.00002)
#achsenabschnitt = ufloat(-2.65400, 0.04789)

for i in range(10):
    gaussenergien[i] = gaussenergien[i] * 0.40298 - 2.654
    i = i + 1

#effizienz = [32.1, 22.94, 14.16, 15.31, 12.62, 6.02, 5.76, 3.03, 3.83, 3.79, 2.52]
gausseffizienz = [0.0672, 0.0384, 0.0310, 0.0263, 0.0104, 0.0092, 0.00696, 0.00569, 0.00571, 0.00364]
gausseffizienzfehler = [0.0025, 0.0007, 0.0022, 0.0015, 0.0006, 0.0005, 0.00022, 0.00032, 0.00017, 0.00014]


#plt.plot(energien, effizienz, 'r.')
#plt.savefig('effizienz.pdf')
#plt.clf()
plt.errorbar(gaussenergien, gausseffizienz, yerr=gausseffizienzfehler, fmt='o', markersize=2, linewidth=1, label='berechnete Effizienzen')

#jetzt kommt Regression, bis hierhin müsste alles gut sein

def ersterfit(energie, a1, b1):
    return a1 * energie**b1

params, covariance_matrix = curve_fit(ersterfit, gaussenergien, gausseffizienz)

errors = np.sqrt(np.diag(covariance_matrix))

print(params[0], errors[0])
print(params[1], errors[1])

x_plot = np.linspace(200, 1500, 10000)

plt.plot(x_plot, ersterfit(x_plot, params[0], params[1]), label='Fit')
plt.xlabel('Energie in keV')
plt.ylabel('Effizienz')
plt.legend()

plt.savefig('gausseffizienz.pdf')

plt.clf()


plt.yscale('log')
plt.xscale('log')
plt.errorbar(gaussenergien, gausseffizienz, yerr=gausseffizienzfehler, fmt='o', markersize=2, linewidth=1)
plt.plot(x_plot, ersterfit(x_plot, params[0], params[1]))

plt.savefig('geffizienzlog.pdf')

#def zweiterfit(energie2, a2, b2, c):
#    return a2 * np.exp(-b2 * energie2) + c
#
#params, covariance_matrix = curve_fit(zweiterfit, gaussenergien, gausseffizienz, p0=[6.26, 1, 726.64])
#
#errors = np.sqrt(np.diag(covariance_matrix))
#
#print(params[0], errors[0])
#print(params[1], errors[1])
#print(params[2], errors[2])
#
#x_plot = np.linspace(200, 3600, 10000)
#
#plt.plot(x_plot, zweiterfit(x_plot, params[0], params[1], params[2]))
