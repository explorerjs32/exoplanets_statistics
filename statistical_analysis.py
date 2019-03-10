import numpy  as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def avg_nightly_observations(MJD, activity_index, activity_index_errors):
    avg_ha_index = []
    avg_ha_index_errors = []
    avg_mjd = []

    for i in MJD.astype(int):
        same = np.where(MJD.astype(int) == i)[0]

        if len(same) == 1:

            if i not in avg_mjd:
                avg_mjd.append(i)
                avg_ha_index.append(activity_index[same][0])
                avg_ha_index_errors.append(activity_index_errors[same][0])              
                
        if len(same) > 1:
        
            if i not in avg_mjd:
                avg_mjd.append(i)
                avg_ha_index.append(np.average(activity_index[same]))
                avg_ha_index_errors.append(np.std(activity_index[same])/np.sqrt(len(activity_index[same])))
                
            else:
                pass
    
    return np.asarray(avg_ha_index), np.asarray(avg_ha_index_errors), np.asarray(avg_mjd)

def bin_data(index, error, date, start, end):

    binned_index = []
    binned_index_error = []
    binned_date = []

    for i in range(len(date)):

        if date[i] >= start and date[i] <= end:
            binned_index.append(index[i])
            binned_index_error.append(error[i])
            binned_date.append(date[i])

    return np.asarray(binned_index), np.asarray(binned_index_error), np.asarray(binned_date)

def extract_binned_data(avg_index, avg_index_error, avg_date):

    start_date = avg_date[0]
    binned_index = []
    binned_mjd = []

    for i in range(len(avg_date)):
        end_date = 150. + start_date
        b = bin_data(avg_index, avg_index_error, avg_date, start_date, end_date)

        if len(b[0]) >= 3:
            binned_index.append(b[0])
            binned_mjd.append(b[2])

        elif start_date >= avg_date[-1]:
            break
        
        start_date = end_date

    return binned_index, binned_mjd

def Fit_Data(x, a, b, c):
    return (a*x**2.) + b*x + c

#################################################################       LONG-TERM VARIABILITY        ##################################################
# Import the H alpha variavility data and plot it
ha_data = np.genfromtxt('/media/fmendez/Seagate_Portable_Drive/Research/Research_Data/HARPS/Ha_index/HD26965_Ha_Variability.dat')

ha_index = ha_data[:,0]
ha_index_error = ha_data[:,1]
mjd = ha_data[:,3]
'''
plt.title(r'$H_{\alpha}$ Index Variability')
plt.xlabel('MJD')
plt.ylabel(r'$H_{\alpha}$ Index')
plt.errorbar(mjd, ha_index, yerr=ha_index_error, fmt='ko')
plt.show()
'''
# Average the nightly observations and plot it
avg_ha_data = avg_nightly_observations(mjd, ha_index, ha_index_error)
'''
plt.title(r'Nightly Averaged $H_{\alpha}$ Index Variability')
plt.xlabel('MJD')
plt.ylabel(r'$H_{\alpha}$ Index')
plt.errorbar(avg_ha_data[2], avg_ha_data[0], yerr=avg_ha_data[1], fmt='ko')
plt.show()
'''
# Bin the nightly average data
binned_ha_index, binned_mjd = extract_binned_data(avg_ha_data[0], avg_ha_data[1], avg_ha_data[2])
        
# Calculate the mean Ha index, MJD, and error of each bin
avg_binned_ha_index = np.zeros(len(binned_ha_index))
avg_binned_mjd = np.zeros(len(binned_ha_index))
binned_ha_index_error = np.zeros(len(binned_ha_index))

for i in range(len(binned_ha_index)):
    avg_binned_ha_index[i] = np.mean(binned_ha_index[i])
    avg_binned_mjd[i] = np.mean(binned_mjd[i])
    binned_ha_index_error[i] = np.std(binned_ha_index[i])/np.sqrt(len(binned_ha_index[i]))

# Calculate the peak-to-peak variation and standard deviation of the binned data
ha_biined_data_standard_deviation = np.std(avg_binned_ha_index)
ha_peak_to_peak_variation = max(avg_binned_ha_index) - min(avg_binned_ha_index)

# Fit the binned ha data and plot the data with the fit
popt, pcov = curve_fit(Fit_Data, avg_binned_mjd, avg_binned_ha_index)
'''
plt.title(r'Long-Term $H_{\alpha}$ Index Variability')
plt.xlabel('MJD')
plt.ylabel(r'$H_{\alpha}$ Index')
plt.plot(np.linspace(min(avg_binned_mjd), max(avg_binned_mjd), 1000.),
         Fit_Data(np.linspace(min(avg_binned_mjd), max(avg_binned_mjd), 1000), *popt), 'k--')
plt.plot(avg_ha_data[2], avg_ha_data[0], 'ko', ms=1, label = 'Nightly Averaged Data')
plt.errorbar(avg_binned_mjd, avg_binned_ha_index, yerr=binned_ha_index_error, fmt='ro', capsize=3, label='Bined Data')
plt.text(57200, 0.4365, r'$\sigma=$ %5.4f' %float(ha_biined_data_standard_deviation))
plt.text(57200, 0.437, r'$\Delta=$ %5.4f' %float(ha_peak_to_peak_variation))
plt.legend()
plt.show()
'''
# Calculate the F-ratio and P-value of the Ha index
ha_f_ratio = (np.std(avg_binned_ha_index)/np.mean(binned_ha_index_error))**2.
ha_p_value = 0.011687 # from a p-value calculator online
''' 
print ('H alpha Long-Term Variability:\n', '- F-ratio: ', ha_f_ratio, '\n- Num/Den DOF: ', len(avg_binned_ha_index) -1, \
      '\n- P-value: ', ha_p_value)
'''
# Import the Sdium Index variability data and plot it
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
# Average the nightly observationsand plot it
avg_na_data = avg_nightly_observations(mjd, na_index, na_index_error)
'''
plt.title('Nightly Averaged Na Index Variability')
plt.xlabel('MJD')
plt.ylabel('Na Index')
plt.errorbar(avg_na_data[2], avg_na_data[0], yerr=avg_na_data[1], fmt='ko')
plt.show()
'''
# Bin the nightly average data 
binned_na_index, binned_mjd = extract_binned_data(avg_na_data[0], avg_na_data[1], avg_na_data[2])

# Calculate the mean Na index, MJD, and error of each bin
avg_binned_na_index = np.zeros(len(binned_na_index))
avg_binned_mjd = np.zeros(len(binned_na_index))
binned_na_index_error = np.zeros(len(binned_na_index))

for i in range(len(binned_na_index)):
    avg_binned_na_index[i] = np.mean(binned_na_index[i])
    avg_binned_mjd[i] = np.mean(binned_mjd[i])
    binned_na_index_error[i] = np.std(binned_na_index[i])/np.sqrt(len(binned_na_index[i]))

# Calculate the peak-to-peak variation and standard deviation of the binned data
na_biined_data_standard_deviation = np.std(avg_binned_na_index)
na_peak_to_peak_variation = max(avg_binned_na_index) - min(avg_binned_na_index)

# Fit the binned ha data and plot the data with the fit
popt, pcov = curve_fit(Fit_Data, avg_binned_mjd, avg_binned_na_index)

plt.title('Long-Term Na Index Variability')
plt.xlabel('MJD')
plt.ylabel('Na Index')
plt.plot(np.linspace(min(avg_binned_mjd), max(avg_binned_mjd), 1000.),
         Fit_Data(np.linspace(min(avg_binned_mjd), max(avg_binned_mjd), 1000), *popt), 'k--')
plt.plot(avg_na_data[2], avg_na_data[0], 'ko', ms=1, label = 'Nightly Averaged Data')
plt.errorbar(avg_binned_mjd, avg_binned_na_index, yerr=binned_na_index_error, fmt='ro', label='Bined Data')
plt.text(57200, 0.2057, r'$\sigma=$ %5.4f' %float(na_biined_data_standard_deviation))
plt.text(57200, 0.2059, r'$\Delta=$ %5.4f' %float(na_peak_to_peak_variation))
plt.legend()
plt.show()

# Calculate the F-ratio and P-value of the Ha index
na_f_ratio = (np.std(avg_binned_na_index)/np.mean(binned_na_index_error))**2.
na_p_value = 0.039467 # from a p-value calculator online

print ('Sodium Long-Term Variability:\n', '- F-ratio: ', na_f_ratio, '\n- Num/Den DOF: ', len(avg_binned_na_index) -1, \
      '\n- P-value: ', na_p_value)





    
    
        
    
    

    






















