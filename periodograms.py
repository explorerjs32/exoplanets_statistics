import numpy as np
import matplotlib.pylab as plt
from PyAstronomy.pyTiming import pyPeriod

##################################################################      PERIODOGRAM ANALYSIS        #########################################################

# Import the H alpha data and plot it
ha_data = np.genfromtxt('/media/fmendez/Seagate_Portable_Drive/Research/Research_Data/HARPS/Ha_index/HD26965_Ha_Variability.dat')

ha_index = ha_data[:,0]
ha_index_error = ha_data[:,1]
mjd = ha_data[:,2]

plt.title(r'$H_{\alpha}$ Index Variability')
plt.xlabel('MJD')
plt.ylabel(r'$H_{\alpha}$ Index')
plt.errorbar(mjd, ha_index, yerr=ha_index_error, fmt='ko')
plt.show()

# Create a GLS Periodogram of the H alpha data
ha_gls = pyPeriod.Gls((mjd, ha_index, ha_index_error))

ha_gls.info()
'''
plt.title(r'$H_{\alpha}$ GLS Periodogram')
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.plot(1./ha_gls.freq, ha_gls.power, 'k')
plt.show()
'''
# Import the Na data and plot it
na_data = np.genfromtxt('/media/fmendez/Seagate_Portable_Drive/Research/Research_Data/HARPS/Na_index/HD26965_Na_Variability.dat')

na_index = na_data[:,0]
na_index_error = na_data[:,1]
mjd = na_data[:,2]
'''
plt.title('Na Index Variability')
plt.xlabel('MJD')
plt.ylabel('Na Index')
plt.errorbar(mjd, na_index, yerr=na_index_error, fmt='ko')
plt.show()
'''
# Create a GLS Periodogram of the H alpha data
na_gls = pyPeriod.Gls((mjd, na_index, na_index_error))

#na_gls.info()
'''
plt.title('Na GLS Periodogram')
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.plot(1./na_gls.freq, na_gls.power, 'k')
plt.show()
'''
















