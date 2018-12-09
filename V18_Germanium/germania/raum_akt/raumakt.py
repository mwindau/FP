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

#abstand = 732
#radius = 22.5

#a = 1/2 * (1 - (abstand)/np.sqrt(abstand**2 + radius**2))

def raumwinkel(a, r):
    return 1/2 * (1 - (a)/(np.sqrt(a**2 + r**2)))

meins = raumwinkel(73.2, 22.5)
#stevens = raumwinkel(881, 27.5)

print("mein Raumwinkel: ", meins)
#print("Stevens Raumwinkel: ", stevens)

a0meins = ufloat(4130, 60)
thalbmeins = ufloat(4943, 5)
tjetztmeins = ufloat(6618, 0)

def aktuelleaktivitaet(a0, thalb, tjetzt):
    return a0 * unp.exp(-(tjetzt * unp.log(2))/(thalb))

aktivitaet = aktuelleaktivitaet(a0meins, thalbmeins, tjetztmeins)
#aktivitaet = aktuelleaktivitaet(4130, 4943, 6618)

print("Aktivität am Messtag: ", aktivitaet)




















