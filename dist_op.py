
from pacal import *
import matplotlib.pyplot as plt

dL = UniformDistr(1,3)
L0 = UniformDistr(9,11)
dT = NormalDistr(1,1)
K = dL / (L0 * dT)

K.plot()
show()
