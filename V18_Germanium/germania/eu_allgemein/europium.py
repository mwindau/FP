import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


a = np.genfromtxt('europium.txt')

c = len(a)

b = np.arange(1, 8192)

def gauss (x, mu, sigma):
    return 1/(np.sqrt(2 * np.pi * sigma**2)) * np.exp(-1/2 * (x - mu)**2/sigma**2)



# normaler Plot des Spektrums


plt.xlim(40, 3700)
#pos, properties = find_peaks(a, prominence=59)
plt.plot(b, a)
#plt.plot(b[pos], a[pos], 'r.')
plt.xlabel('Kanalnummer')
plt.ylabel('Counts')
plt.savefig('europium.pdf')
plt.clf()

## normaler Plot oben abgeschnitten
#
#plt.xlim(40, 3700)
#plt.ylim(-100, 2000)
#plt.plot(b, a)
#plt.savefig('europiumschnitto.pdf')
#plt.clf()


# logarithmischer Plot

plt.xlabel('Kanalnummer')
plt.ylabel('Counts')
plt.xlim(40, 3700)
plt.yscale('log')
plt.plot(b, a)
plt.legend()
plt.savefig('europilog.pdf')
plt.clf()

# logarithmischer Plot mit Schnitt vorne

#plt.xlim(300, 3700)
#plt.xscale('log')
#plt.plot(b, a)
#plt.savefig('europilogschnittv.pdf')
#plt.clf()
#
## logarithmischer Plot Schnitt oben und vorne
#
#plt.xlim(300, 3700)
#plt.ylim(-100, 2000)
#plt.xscale('log')
#plt.plot(b, a)
#plt.savefig('europilogschnittvo.pdf')
#plt.clf()

## zoom auf den ersten Peak
#
#plt.xlim(300, 320)
#plt.plot(b, a)
#c = gauss(b, mu, sigma)
#params1, cov1 = curve_fit(gauss, b, c)
#plt.plot(b, f1(b, *noms(params1))
##plt.plot(b, gauss(b, *popt), 'r-')
##plt.show()
#plt.clf()
#
## zoom auf den zweiten Peak
#
#plt.xlim(600, 630)
#plt.ylim(-100, 1000)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak244.pdf')
#plt.clf()
#
## zoom auf den dritten Peak
#
#plt.xlim(850, 875)
#plt.ylim(-100, 2000)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak344.pdf')
#plt.clf()
#
## zoom auf den vierten Peak
#
#plt.xlim(1010, 1045)
#plt.ylim(-100, 400)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak411.pdf')
#plt.clf()
#
## zoom auf den fünften Peak
#
#plt.xlim(1090, 1125)
#plt.ylim(-100, 300)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak443.pdf')
#plt.clf()
#
## zoom auf den sechsten Peak
#
#plt.xlim(1920, 1960)
#plt.ylim(-100, 500)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak778.pdf')
#plt.clf()
#
## zoom auf den siebten Peak
#
#plt.xlim(2145, 2175)
#plt.ylim(-100, 150)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak867.pdf')
#plt.clf()
#
## zoom auf den achten Peak
#
#plt.xlim(2380, 2420)
#plt.ylim(-100, 300)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak964.pdf')
#plt.clf()
#
## zoom auf den neunten Peak
#
#plt.xlim(2675, 2730)
#plt.ylim(-100, 200)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak1085.pdf')
#plt.clf()
#
## zoom auf den zehnten Peak
#
#plt.xlim(2740, 2790)
#plt.ylim(-100, 200)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak1112.pdf')
#plt.clf()
#
## zoom auf den elften Peak
#
#plt.xlim(3480, 3520)
#plt.ylim(-100, 200)
#plt.plot(b, a)
##plt.show()
#plt.savefig('peak1408.pdf')
#plt.clf()
































































#platz
