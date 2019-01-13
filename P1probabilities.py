# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from numpy import exp, sqrt, pi

mu_u = .5
sigma_u = .16
u = np.linspace(0, 1, 100)
plt.plot(u, mlab.normpdf(u, mu_u, sigma_u))
plt.show()

r = np.linspace(0, 5, 500)

pdf_r = 1/sqrt(2*pi*sigma_u**2)*exp(-.5*(.5-exp((-.5*r**2)))**2/sigma_u**2)*r*exp((-.5*r**2))

plt.plot(r,pdf_r)
plt.show()


mu_v = .5
sigma_v = .16
v = np.linspace(0, 1, 100)
plt.plot(u, mlab.normpdf(v, mu_v, sigma_v))
plt.show()


theta = np.linspace(0, 2*pi, 100)

pdf_theta = 1/sqrt(2*pi*sigma_v**2)*exp(-.5*(.5/pi*theta-.5)**2/sigma_v**2)/(2*pi)

plt.plot(theta,pdf_theta)
plt.show()

