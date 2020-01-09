import numpy as np 
import matplotlib.pyplot as plt 

a, b1, b2, b3 = np.genfromtxt('bfeld.txt', unpack=True)

b_mittel = np.array([0])

for i in range(18):
    a = b1[i]
    b = b2[i]
    c = b3[i]
    neu = np.mean(a, b, c)
    b_mittel = np.append(b_mittel, neu)


plt.plot(a, b_mittel, 'b.')
plt.show()

