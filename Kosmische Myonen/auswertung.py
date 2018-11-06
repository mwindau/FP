import numpy as np
from scipy.stats import sem
from uncertainties import ufloat
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit

#Verzögerungszeit:
T_vz, n_vz_10 = np.genfromtxt('verzoegerung.txt', unpack=True)
#n_vz = n_vz_10/10
n_vz_err = np.sqrt(n_vz_10)

P = unp.uarray([n_vz_10],[n_vz_err])

def plateau_regression(x,b):
    return 0*x + b

plateau_params, plateau_covariance = curve_fit(plateau_regression, T_vz[16:31], n_vz_10[16:31], sigma = n_vz_err[16:31], absolute_sigma= True )
plateau_errors = np.sqrt(np.diag(plateau_covariance))
hoehe_plateau = ufloat(plateau_params[0],plateau_errors[0])

print("Höhe des Plateau: ", hoehe_plateau)

def flanken_regression(x,a,b):
    return a*x + b

flanke_l_params, flanke_l_covariance = curve_fit(flanken_regression, T_vz[6:17], n_vz_10[6:17], sigma = n_vz_err[6:17], absolute_sigma= True)
flanke_l_errors = np.sqrt(np.diag(flanke_l_covariance))
flanke_r_params, flanke_r_covariance = curve_fit(flanken_regression, T_vz[31:39], n_vz_10[31:39], sigma = n_vz_err[31:39], absolute_sigma= True)
flanke_r_errors = np.sqrt(np.diag(flanke_r_covariance))

flanke_l_a = ufloat(flanke_l_params[0],flanke_l_errors[0])
flanke_l_b = ufloat(flanke_l_params[1],flanke_l_errors[1])
flanke_r_a = ufloat(flanke_r_params[0],flanke_r_errors[0])
flanke_r_b = ufloat(flanke_r_params[1],flanke_r_errors[1])

print("Flankenparamter a für linke Flanke: ", flanke_l_a,"Flankenparamter b für linke Flanke: ", flanke_l_b)
print("Flankenparamter a für rechte Flanke: ", flanke_r_a,"Flankenparamter b für rechte Flanke: ", flanke_r_b)

T_vz_l = (hoehe_plateau/2-flanke_l_b)/flanke_l_a
T_vz_r = (hoehe_plateau/2-flanke_r_b)/flanke_r_a

print("Halbwertzeitabstand links: ", T_vz_l)
print("Halbwertzeitabstand rechts: ", T_vz_r)

T_vz_haelfte = T_vz_r - T_vz_l

print("Halbwertzeitabstand: ", T_vz_haelfte)

tfit = np.linspace(-20,35,100000)

#plt.errorbar(T_vz,n_vz_10,xerr=0,yerr=n_vz_err,fmt='x', label=r'Messwerte')
#plt.plot(tfit, plateau_regression(tfit,*plateau_params), '--',label=r'Plateau-Fit')
#plt.plot(tfit, flanken_regression(tfit,*flanke_l_params), '--',label=r'Fit der linken Flanke')
#plt.plot(tfit, flanken_regression(tfit,*flanke_r_params), '--',label=r'Fit der rechten Flanke')
#plt.xlabel(r'$T_{\mathrm{VZ}}\;/\;\mathrm{ns}$')
#plt.ylabel(r'$N(T_{\mathrm{VZ}}$)')
#plt.ylim(-2,320)
#
#plt.legend(loc="best")
#plt.savefig('verzoegerungregression.pdf')
#plt.clf()

# plt.plot(T_vz,n_vz_10, 'x',label='Messwerte')
#plt.errorbar(T_vz,n_vz_10, xerr=0,yerr=n_vz_err,fmt='x', capsize=3, label='Messwerte')
#plt.xlabel(r'$T_{\mathrm{VZ}}\;/\;\mathrm{s}$')
#plt.ylabel(r'$N(T_{\mathrm{VZ}})$')
#plt.legend(loc="best")
#plt.savefig('verzoegerung.pdf')
#plt.clf()
#################################################################################
#Vielkanalanalysator:
kanal, n_dp, zeit_kanal = np.genfromtxt('doppelimpuls.txt', unpack=True)

def zeitregression(x,a,b):
    return a*x + b

zeitkanalparameter, zeitkanalparameter_covariance = curve_fit(zeitregression,kanal,zeit_kanal)
zeitkanal_errors = np.sqrt(np.diag(zeitkanalparameter_covariance))
a_zeitkanal = ufloat(zeitkanalparameter[0],zeitkanal_errors[0])
b_zeitkanal = ufloat(zeitkanalparameter[1],zeitkanal_errors[1])

print("a: ", a_zeitkanal)
print("b: ", b_zeitkanal)
##################################################################################
#Lebensdauer:
N_start = 5345056
N_stopp = 10995
T_messung = 158215
T_such = 13*10**(-6)

R = N_start/T_messung
N_such = R*T_such
W1 = N_such*np.exp(-N_such)
U1gesamt = W1*N_start
U1 = U1gesamt/511

print("R: ", R)
print("W: ", W1)
print("U: ",U1gesamt)
print("U1: ", U1)

N_muon = np.genfromtxt('myonen.txt', unpack=True)
N_muon_err = np.sqrt(N_muon)

kanalmuon = np.linspace(0,511,512)
indexmuon0 = [0,1,2,421,426,463,472,498,499,500,501,502,503,504,505,506,507,507,508,509,510,511]
kanalmuon_neu = np.delete(kanalmuon, indexmuon0)
zeitmuon = zeitregression(kanalmuon_neu,a_zeitkanal,b_zeitkanal)
zeitmuon_neu = unp.nominal_values(zeitmuon)
#zeitmuon2 = np.round(zeitmuon_neu, 3)
zeitmuon3 = np.float64(zeitmuon_neu)
zeitmuon4 = np.round(zeitmuon_neu, 5)
zeitmuon4float64 = np.float64(zeitmuon4)
zeitmuon_neu_neu = np.array(zeitmuon_neu, dtype=np.float128)

#print(zeitmuon_neu)
#print(np.round(zeitmuon_neu, 3))
#print(N_muon)


def muonfit(x,N0,l,U):
    return np.float64(N0 * np.exp(-l*x) + U)


muon_parameter,muon_covariance = curve_fit(muonfit,zeitmuon_neu_neu,N_muon,sigma = N_muon_err, absolute_sigma = True)
muon_parameter_errors = np.sqrt(np.diag(muon_covariance))
N0 = ufloat(muon_parameter[0],muon_parameter_errors[0])
lambd = ufloat(muon_parameter[1],muon_parameter_errors[1])
U2 = ufloat(muon_parameter[2],muon_parameter_errors[2])

print("N0: ", N0)
print("lambda: ", lambd)
print("U2: ", U2)
print("tau: ", 1/lambd)

t_muon = np.linspace(-2,13,10000)

#print(zeitmuon_neu_neu)
#print(len(zeitmuon_neu_neu))
#print(len(N_muon))


plt.errorbar(zeitmuon_neu_neu,N_muon,xerr=0,yerr=N_muon_err, fmt='kx', label= r'Messwerte')
plt.plot(t_muon,muonfit(t_muon,*muon_parameter), label=r'Fit')
plt.xlabel(r'$t\;/\;\mathrm{ns}$')
plt.ylabel(r'$N(t)$')
plt.legend(loc="best")
plt.savefig('blabla.pdf')
plt.clf()
