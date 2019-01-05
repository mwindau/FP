import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values

######################################################################

def g1g2(r1,r2,L):
    return 1 - L/r1 - L/r2 + L**2/(r2*r1)

r_mirror_1 = 1
r_mirror_2 = 1.40
r_flat = np.inf

L_linspace = np.linspace(0,3, 100000)

plt.plot(L_linspace, g1g2(r_mirror_1,r_mirror_1,L_linspace), label='2x r_1 = 1000mm')
plt.plot(L_linspace, g1g2(r_mirror_2,r_mirror_2,L_linspace), label='2x r_2 = 1400mm')
plt.plot(L_linspace, g1g2(r_flat,r_mirror_2,L_linspace), label='r_2 = 1400mm und r_3 = $\inf$')
plt.axhspan(0,1,color='r', alpha=0.2, lw=0)
plt.ylim(-0.5,1.5)
plt.legend()
plt.grid()
#plt.show()
plt.savefig('vorbereitung.pdf')
plt.clf()

#################################################
#Stabilitätsmessung
#################################################

#Konkav-Konkav
print('\n\n\n-----------------------------------------------------------\n','Stabilitätsbedinung Konkav/konkav', '\n', '-----------------------------------------------------------\n\n\n')
L_kk,I_kk = np.genfromtxt('konkavkav.txt', unpack='True')
L_kk_accepted = L_kk[4:18]
I_kk_accepted = I_kk[4:18]

##curvefit
def fit_konkavkav(L,a,b,c):
    return a*L**2 + b*L + c
params_konkavkav, cov_konkavkav = curve_fit(fit_konkavkav, L_kk_accepted, I_kk_accepted)
error_konkavkav = np.sqrt(np.diag(cov_konkavkav))
print('a = {:.3f} ± {:.3f} \muA/cm^2'.format(params_konkavkav[0], error_konkavkav[0]))
print('b = {:.3f} ± {:.3f} \muA/cm'.format(params_konkavkav[1], error_konkavkav[1]))
print('c = {:.3f} ± {:.3f} \muA'.format(params_konkavkav[2], error_konkavkav[2]))

x_kk = np.linspace(L_kk_accepted[0]-1, L_kk_accepted[-1]+1,1000)
##Plot
plt.plot(L_kk_accepted,I_kk_accepted,'rx', label='Messdaten für Konkav-Konkav')
plt.plot(x_kk,fit_konkavkav(x_kk,params_konkavkav[0], params_konkavkav[1], params_konkavkav[2]), label='Fit')
plt.xlabel('L in cm')
plt.ylabel('I in $\mu$A')
plt.legend()
#plt.show()
plt.savefig('stabilitaet_konkavkonkav.pdf')
plt.clf()

#Planar-Konkav
print('\n\n\n-----------------------------------------------------------\n','Stabilitätsbedinung Planar/konkav', '\n', '-----------------------------------------------------------\n\n\n')
L_pk,I_pk = np.genfromtxt('planarkonkav.txt', unpack='True')

##curvefit
def fit_planarkonkav(L,a,b):
    return a*L + b
params_planarkonkav, cov_planarkonkav = curve_fit(fit_planarkonkav, L_pk, I_pk)
error_planarkonkav = np.sqrt(np.diag(cov_planarkonkav))
print('a = {:.3f} ± {:.3f} \muA/cm'.format(params_planarkonkav[0], error_planarkonkav[0]))
print('b = {:.3f} ± {:.3f} \muA'.format(params_planarkonkav[1], error_planarkonkav[1]))

x_pk = np.linspace(L_pk[0]-1, L_pk[-1]+1,1000)
##Plot
plt.plot(L_pk,I_pk,'rx', label='Messdaten für Planar-Konkav')
plt.plot(x_pk,fit_planarkonkav(x_pk,params_planarkonkav[0], params_planarkonkav[1]), label='Fit')
plt.xlabel('L in cm')
plt.ylabel('I in $\mu$A')
plt.legend()
#plt.show()
plt.savefig('stabilitaet_planarkonkav.pdf')
plt.clf()


print('\n###################################################################\n')
##############################################################################################
#Polarisation
##########################################################################################
print('\n\n\n-----------------------------------------------------------\n','Polarisation', '\n', '-----------------------------------------------------------\n\n\n')

phi_deg, I_pol = np.genfromtxt('polarisation.txt', unpack='True')
phi_rad = phi_deg * 2*np.pi/360

##curvefit
def fit_pol(phi,I_0,phi_0):
    return I_0 * np.cos(phi-phi_0)**2
params_pol, cov_pol = curve_fit(fit_pol, phi_rad, I_pol)
errors_pol = np.sqrt(np.diag(cov_pol))
print('I_0 = {:.3f} ± {:.3f} \muA'.format(params_pol[0], errors_pol[0]))
print('phi_0 = {:.3f} ± {:.3f} rad'.format(params_pol[1], errors_pol[1]))

phi = np.linspace(phi_rad[0]-1, phi_rad[-1]+1,1000)
##Plot
plt.plot(phi_rad,I_pol,'rx', label='Messdaten')
plt.plot(phi,fit_pol(phi,params_pol[0], params_pol[1]), label='Fit')
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
           [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])
plt.xlabel('$\phi$ in rad')
plt.ylabel('I in $\mu$A')
plt.legend()
#plt.show()
plt.savefig('polarisation.pdf')
plt.clf()


print('\n###################################################################\n')
#########################################################################
#Modenmessung
#########################################################################
print('\n\n\n-----------------------------------------------------------\n','Modenmessung', '\n', '-----------------------------------------------------------\n\n\n')

####################
#Grundmode
L_grund, I_grund = np.genfromtxt('grundmode.txt', unpack='True')

##curvefit
def fit_grund(L,I_0,d_0,w):
    return I_0 * np.exp(-2 * ((L-d_0) / w)**2)
params_grund, cov_grund = curve_fit(fit_grund, L_grund, I_grund)
errors_grund = np.sqrt(np.diag(cov_grund))
print('I_0 = {:.3f} ± {:.3f} \muA'.format(params_grund[0], errors_grund[0]))
print('d_0 = {:.3f} ± {:.3f} mm'.format(params_grund[1], errors_grund[1]))
print('w = {:.3f} ± {:.3f} mm'.format(params_grund[2], errors_grund[2]))

x_grund = np.linspace(L_grund[0]-1, L_grund[-1]+1,1000)
##Plot
plt.plot(L_grund,I_grund,'rx', label='Messdaten')
plt.plot(x_grund,fit_grund(x_grund,params_grund[0], params_grund[1], params_grund[2]), label='Fit')
plt.xlabel('L in mm')
plt.ylabel('I in $\mu$A')
plt.legend()
#plt.show()
plt.savefig('grundmode.pdf')
plt.clf()

################################
#Erste Mode
L_erste, I_erste = np.genfromtxt('erstemode.txt', unpack='True')

##curvefit
def fit_erste(L,I_01,d_01,w1,I_02,d_02,w2):
    return I_01 * np.exp(-2 * ((L-d_01) / w1)**2) + I_02 * np.exp(-2 * ((L-d_02) / w2)**2)

###Startwerte:
I_01_max = 91.7
I_02_max = 78.2
L_01_max = -14.5
L_02_max = 6.8

params_erste, cov_erste = curve_fit(fit_erste, L_erste, I_erste, p0=[I_01_max,L_01_max,1,I_02_max,L_02_max,1])
errors_erste = np.sqrt(np.diag(cov_erste))
print('I_01 = {:.3f} ± {:.3f} nA'.format(params_erste[0], errors_erste[0]))
print('d_01 = {:.3f} ± {:.3f} mm'.format(params_erste[1], errors_erste[1]))
print('w_1 = {:.3f} ± {:.3f} mm'.format(params_erste[2], errors_erste[2]))
print('I_02 = {:.3f} ± {:.3f} nA'.format(params_erste[3], errors_erste[3]))
print('d_02 = {:.3f} ± {:.3f} mm'.format(params_erste[4], errors_erste[4]))
print('w_2 = {:.3f} ± {:.3f} mm'.format(params_erste[5], errors_erste[5]))

x_erste = np.linspace(L_erste[0]-1, L_erste[-1]+1,1000)
##Plot
plt.plot(L_erste,I_erste,'rx', label='Messdaten')
plt.plot(x_erste,fit_erste(x_erste,params_erste[0], params_erste[1], params_erste[2],params_erste[3], params_erste[4], params_erste[5]), label='Fit')
plt.xlabel('L in mm')
plt.ylabel('I in nA')
plt.legend()
#plt.show()
plt.savefig('erstemode.pdf')
plt.clf()

######################################################################
print('\n###################################################################\n')
#########################################################################
#Wellenlänge
#########################################################################
print('\n\n\n-----------------------------------------------------------\n','Wellenlängenmessung', '\n', '-----------------------------------------------------------\n\n\n')

dist_small = np.array([4.2,4.25]) #Abstand zwischen den Maxima der ersten Ordnung zur nullten Ordnung
dist_large = 67                   #Abstand zwischen Gitter und Blende
a = 1/1000                        #Gitterkonstante
k = 1                             #Ordnung der ersten Maxima

dist_small_mean = np.mean(dist_small)
dist_small_std = np.std(dist_small)
dist_small = ufloat(dist_small_mean,dist_small_std)

phi_lambda = unp.arctan(dist_small/dist_large)
lambdaa = a / k * unp.sin(phi_lambda)

print('$\lambda$ =', lambdaa, 'cm')
