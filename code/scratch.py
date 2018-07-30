from __future__ import division
import numpy
import matplotlib.pyplot as plt
from crawler import Crawler
#pip install pycwt
#Conda does not work
import pycwt as wavelet
from pycwt.helpers import find

url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
dat = numpy.genfromtxt(url, skip_header=19)
title = 'NINO3 Sea Surface Temperature'
label = 'NINO3 SST'
units = 'degC'
t0 = 1871.0
dt = 0.25  # In years
N = dat.size
t = numpy.arange(0, N) * dt + t0

p = numpy.polyfit(t - t0, dat, 10)
dat_notrend = dat - numpy.polyval(p, t - t0)
std = dat_notrend.std()  # Standard deviation
var = std ** 2  # Variance
dat_norm = dat_notrend / std  # Normalized dataset
x = [x for x in range(len(dat))]
# plt.plot(x, dat_norm, x, dat)
# plt.show()

c = Crawler()  
filename = "2018/hourly_1530608400.json"
#closing price
dat = c.json2List( filename, 'close')
title = 'BTC closing price ending 3-7-2018'
label = 'BTC Close'
units = 'USD'
mean = dat.mean()
std = dat.std()
var = std ** 2  # Variance
dat_norm = (dat - mean) / std  # Subtracting mean

x = [x for x in range(len(dat))]
plt.plot(x, dat)
plt.show()
