import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#imax = 1.56
#imin = 0.86
#
#print("kontrast: ", ((imax-imin)/(imax+imin)))

winkel, imax, imin, kontrast_0 = np.genfromtxt("kontrast.txt", unpack = True)

plt.plot(winkel, kontrast_0)

def kontrast (theta, a, b, theta_0):
    return a * np.absolute(np.sin(2*theta+theta_0))+b

params, covariance = curve_fit(kontrast, winkel, kontrast_0, p0=(0.8, 0.1, 1))

print(params)

#x_plot = np.linspace(0, 360, )
#
#plt.plot(x_plot, kontrast(x_plot, 0.8, 0.1, 0))


plt.savefig('kontrast.pdf')
