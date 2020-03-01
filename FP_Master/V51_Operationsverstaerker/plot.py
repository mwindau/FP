import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties import correlated_values
from scipy.optimize import curve_fit
import scipy.constants as const


# Data
zerhacker_frequency = 450 #Hz Frequenz des Lichtzerhackers

N2                  = 1.2*10**(18) #in 1/cm^3  / (10**(-6)) #1/m^3 Dotierung Probe 2
N3                  = 2.8*10**(18) # " / (10**(-6)) #1/m^3 Dotierung Probe 3

L1                  = 5.11 #mm 0.00511 #m Länge der undotierten Probe (1. Probe)
L2                  = 1.36 #mm 0.00136 #m Länge der 2. Probe
L3                  = 1.296 #mm 0.001296 #m Länge der 3. Probe

wellenlaengen_mum   = np.array([1.06,1.29, 1.45, 1.72, 1.96, 2.15, 2.34, 2.51, 2.65]) #mikrometer Wellenlaenge der Filter 
wellenlaengen       = 10**(-6) * wellenlaengen_mum


n_array                   = np.array([3.4744,3.4078,3.3820,3.3551,3.3404,3.3323,3.3261,3.3217,3.3187])
n_mean                    = np.mean(n_array)
n_std                     = np.std(n_array)
n                         = unp.uarray(n_mean, n_std)

junk, theta11g, theta11m, theta12g, theta12m = np.genfromtxt('data/probe1.txt', unpack=True)  
junk2, theta21g, theta21m, theta22g, theta22m = np.genfromtxt('data/probe2.txt', unpack=True)
junk3, theta31g, theta31m, theta32g, theta32m = np.genfromtxt('data/probe3.txt', unpack=True)

theta11 = theta11g + theta11m/60
theta12 = theta12g + theta12m/60
theta21 = theta21g + theta21m/60
theta22 = theta22g + theta22m/60
theta31 = theta31g + theta31m/60
theta32 = theta32g + theta32m/60

z_mm, B_mT = np.genfromtxt('data/hall.txt', unpack=True)
B_max = 419 #mT maximum bei z=94mm
###############

# Analysis

def degree_rad(theta):
    theta_rad = theta * np.pi/180
    return theta_rad

def theta_sum(theta_1,theta_2):
    theta = 0.5 * np.abs(theta_1 - theta_2)
    return theta

def m_eff_squared(N, B, n, a):
    return (N*B*const.elementary_charge**3)/(8*np.pi**2*const.epsilon_0*const.speed_of_light**3*n*a)

e0 = const.elementary_charge
print("e0: ", e0)

theta_array = np.array([theta_sum(degree_rad(theta11),degree_rad(theta12)),
                       theta_sum(degree_rad(theta21),degree_rad(theta22)), 
                       theta_sum(degree_rad(theta31),degree_rad(theta32))])
                
theta_array[0] /= L1
theta_array[1] /= L2
theta_array[2] /= L3

delta_theta_2 = np.abs(theta_array[1]-theta_array[0])
delta_theta_3 = np.abs(theta_array[2]-theta_array[0])

#print("n: ", n_array)
#print("\nz in mm: ", z_mm)
#print("\nB in mT: ", B_mT)
#print("\nB_max: ", B_max)
#print("\nn_mean: ", n)
#print("\nTheta11: ", theta11)
#print("\nTheta12: ", theta12)
#print("\nTheta21: ", theta21)
#print("\nTheta22: ", theta22)
#print("\nTheta31: ", theta31)
#print("\nTheta32: ", theta32)
#print("\nTheta1_transf: ", theta_array[0])
#print("\nTheta2_transf: ", theta_array[1])
#print("\nTheta3_transf: ", theta_array[2])
#print("delta_theta_2 = ", delta_theta_2[:])
#print("\ndelta_theta_3 = ", delta_theta_3[:])
#############################

# Curve Fit

def linear_regression(wave_length, A):
    return A * wave_length# + b


#Probe 2
params_2_1, covariance_matrix_2 = curve_fit(linear_regression, wellenlaengen_mum**2, delta_theta_2)
params_2                        = correlated_values(params_2_1, covariance_matrix_2)
#Probe 3
params_3_1, covariance_matrix_3 = curve_fit(linear_regression, wellenlaengen_mum**2, delta_theta_3)
params_3                        = correlated_values(params_3_1, covariance_matrix_3)

#Plot
wave_length_squared_linspace = np.linspace(wellenlaengen_mum[0]-1, wellenlaengen_mum[-1]**2+1, 1000)

##Plot für Probe 2
#plt.plot(wellenlaengen_mum**2, delta_theta_2, 'rx', label='Probe 2')
#plt.plot(wave_length_squared_linspace, linear_regression(wave_length_squared_linspace, params_2_1), label='Fit für 2. Probe')
#plt.xlabel(r'$\lambda^2$ in $\SI{}{\micro\metre}$')
#plt.ylabel(r'$\Delta\Theta$ in 1 / $\SI{}{\milli\metre}$')
#plt.tight_layout()
#plt.legend()
#plt.grid()
##plt.show()
#plt.savefig('build/probe2.pdf')
#plt.clf()
#
##Plot für Probe 3
#plt.plot(wellenlaengen_mum**2, delta_theta_3, 'rx', label='Probe 3')
#plt.plot(wave_length_squared_linspace, linear_regression(wave_length_squared_linspace, params_3_1), label='Fit für 3. Probe')
#plt.xlabel(r'$\lambda^2$ in $\SI{}{\micro\metre}$')
#plt.ylabel(r'$\Delta\Theta$ in 1 / $\SI{}{\milli\metre}$')
#plt.tight_layout()
#plt.legend()
#plt.grid()
##plt.show()
#plt.savefig('build/probe3.pdf')
#plt.clf()

#Plot für Probe 2 und 3
plt.plot(wellenlaengen_mum**2, delta_theta_2, 'rx', label='Probe 2')
plt.plot(wave_length_squared_linspace, linear_regression(wave_length_squared_linspace, params_2_1), label='Fit für 2. Probe')
plt.plot(wellenlaengen_mum**2, delta_theta_3, 'kx', label='Probe 3')
plt.plot(wave_length_squared_linspace, linear_regression(wave_length_squared_linspace, params_3_1), label='Fit für 3. Probe')
plt.xlabel(r'$\lambda^2$ in $\SI{}{\square\micro\metre}$')
plt.ylabel(r'$\Delta\Theta$ in rad / $\SI{}{\milli\metre}$')
plt.tight_layout()
plt.legend()
plt.grid()
#plt.show()
plt.savefig('build/probe23.pdf')
plt.clf()

#############################

# Effektive Masse

m_eff_squared_12 = np.array([
    m_eff_squared(N2*10**6, B_max*10**(-3), n, params_2[0]*10**(15)),
    m_eff_squared(N3*10**6, B_max*10**(-3), n, params_3[0]*10**(15))
    ])

print("\n####################################\n")
print("N2, N3 in 1/cm^3 : ", N2, N3)
print("\n####################################\n")
print("L1, L2, L3 in mm: ", L1, L2, L3)
print("\n####################################\n")
print("wellenlängen in mum und m: ", wellenlaengen_mum, wellenlaengen)
print("\n####################################\n")
print("n: ", n_array)
print("\nn_mean: ", n)
print("\n####################################\n")
print("\nz in mm: ", z_mm)
print("\nB in mT: ", B_mT)
print("\nB_max: ", B_max)
print("\n####################################\n")
print("\nTheta11: ", theta11)
print("\nTheta12: ", theta12)
print("\nTheta21: ", theta21)
print("\nTheta22: ", theta22)
print("\nTheta31: ", theta31)
print("\nTheta32: ", theta32)
print("\n####################################\n")
print("\nTheta1_transf: ", theta_array[0])
print("\nTheta2_transf: ", theta_array[1])
print("\nTheta3_transf: ", theta_array[2])
print("\n####################################\n")
print("\ndelta_theta_2 = ", delta_theta_2[:])
print("\ndelta_theta_3 = ", delta_theta_3[:])
print("\n####################################\n")
print("\nparams_2: ", params_2)
print("\nparams_3: ", params_3)
print("\n####################################\n")
print("\nProbe 2: m_eff = ", (m_eff_squared_12[0])**(1/2))
print("\nProbe 3: m_eff = ", (m_eff_squared_12[1])**(1/2))
print("\n####################################\n")

me = const.electron_mass
print("me = ", me)
print("m*/me = ", m_eff_squared_12**(1/2)/me)





