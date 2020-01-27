
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.random.normal(loc=1., scale=1., size=10000)
y = np.random.normal(loc=2., scale=2., size=10000)
z = x + y

# print (np.mean(x) + np.mean(y), np.sqrt(np.std(x) ** 2 + np.std(y) ** 2))
# print (np.mean(z), np.std(z))

############################

def pdf(x):
  return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

x = np.linspace(-3., 3., 100)
plt.plot(x, pdf(x))
# plt.show()

y =  pdf(0.) - pdf(-100.)
# print (y)

############################

x = norm()
print (x.cdf(0.))

############################
