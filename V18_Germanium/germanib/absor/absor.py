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





inhalt_photopeak = 2560.73
inhalt_kontinuum = 53662.93
d = 3.9
mu_photo = 0.008
mu_konti = 0.37

a = inhalt_kontinuum/inhalt_photopeak

print('Inhalt Photopeak: ', inhalt_photopeak)

print('Inhalt Kontinumm: ', inhalt_kontinuum)

print('Verhältnis: ', a)

print('d =', d)

print('mu Photoeffekt: ', mu_photo)

print('mu Kontinuum: ', mu_konti)

def wkeit(mu):
    return (1 - np.exp(- mu * 3.9))

print('Wahrscheinlichkeit Photoeffekt: ', wkeit(mu_photo))

print('Wahrscheinlichkeit Comptoneffekt: ', wkeit(mu_konti))

print('Verhältnis: ', wkeit(mu_konti)/wkeit(mu_photo))
