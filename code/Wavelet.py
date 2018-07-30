from __future__ import division

import matplotlib.pyplot as plt
<<<<<<< HEAD
import numpy as np
=======
from crawler import Crawler
>>>>>>> parent of 4600d51... Adding Genetic Algorithm
#pip install pycwt
#Conda does not work
import pycwt as wavelet
from pycwt.helpers import find

from crawlers.crawler import Crawler 

# url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
# dat = np.genfromtxt(url, skip_header=19)
c = Crawler()  
filename = "2018/hourly_1530608400.json"
#closing price
dat = np.array(c.json2List( filename, 'close')[:200])
# dat = np.array([x for x in range(500)])
title = 'BTC closing price ending 3-7-2018'
label = 'BTC Close'
units = 'USD'
#initial time set to 0 for now
t0 = 0
dt = 1  # In hours
# Create a time array in years.

N = dat.size
t = np.arange(0, N) * dt + t0
# We write the following code to detrend and normalize the input data by its standard deviation. 
# Sometimes detrending is not necessary and simply removing the mean value is good enough. 
# However, if your dataset has a well defined trend, 
# such as the Mauna Loa CO2 dataset available in the above mentioned website, 
# it is strongly advised to perform detrending. 
# Here, we fit a one-degree polynomial function and then subtract it from the original data.

# p = np.polyfit(t - t0, dat, 1)
# dat_notrend = dat - np.polyval(p, t - t0)
# std = dat_notrend.std()  # Standard deviation
# var = std ** 2  # Variance
# dat_norm = dat_notrend / std  # Normalized dataset

mean = dat.mean()
std = dat.std()
var = std ** 2  # Variance
dat_norm = (dat - mean) / std  # Subtracting mean

mother = wavelet.Morlet(6)
# plt.plot(dat)
# plt.show()
s0 = 2 * dt  # Starting scale. 2 hours
#dj = 1 / 12  # Twelve sub-octaves per octaves
dj = 1 / 24  # Twelve sub-octaves per octaves

J = 7 / dj  # Seven powers of two with dj sub-octaves
alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

power = (np.abs(wave)) ** 2
fft_power = np.abs(fft) ** 2
period = 1 / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95

glbl_power = power.mean(axis=1)
dof = N - scales  # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)

sel = find((period >= 2) & (period < 8))
Cdelta = mother.cdelta
scale_avg = (scales * np.ones((N, 1))).transpose()
scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                             significance_level=0.95,
                                             dof=[scales[sel[0]],
                                                  scales[sel[-1]]],
                                             wavelet=mother)
# Prepare the figure
plt.close('all')
plt.ioff()
figprops = dict(figsize=(11, 8), dpi=72)
fig = plt.figure(**figprops)

# First sub-plot, the original time series anomaly and inverse wavelet
# transform.
ax = plt.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(t, dat, 'k', linewidth=1.5)
ax.set_title('a) {}'.format(title))
ax.set_ylabel(r'{} [{}]'.format(label, units))

# Second sub-plot, the normalized wavelet power spectrum and significance
# level contour lines and cone of influece hatched area. Note that period
# scale is logarithmic.
bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
# bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
#             extend='both', cmap=plt.cm.viridis)
bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
            extend='both')
extent = [t.min(), t.max(), 0, max(period)]
bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                           np.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
bx.set_ylabel('Period (hours)')
#
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels(Yticks)

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, np.log2(period), 'k--')
cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
cx.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
        linewidth=1.)
cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
cx.set_xlabel(r'Power [({})^2]'.format(units))
cx.set_xlim([0, glbl_power.max() + var])
cx.set_ylim(np.log2([period.min(), period.max()]))
cx.set_yticks(np.log2(Yticks))
cx.set_yticklabels(Yticks)
plt.setp(cx.get_yticklabels(), visible=False)

# Fourth sub-plot, the scale averaged wavelet spectrum.
dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(t, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) {}--{} hour scale-averaged power'.format(2, 8))
dx.set_xlabel('Time (hour)')
dx.set_ylabel(r'Average variance [{}]'.format(units))
ax.set_xlim([t.min(), t.max()])

plt.show()                                             
