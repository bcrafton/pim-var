
from scipy import stats
from scipy.stats import norm

rv = norm()
print (rv.cdf(0))

a = norm()
b = norm()
rv2 = a.rvs() + b.rvs()
print (rv2)
# print (rv2.cdf(0))

print (a.rvs())
print (b.rvs())

print (a.rvs())

print (a.rvs())
