import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.signal import hilbert, chirp
import pandas as pd


# import data
data = np.genfromtxt('TEK00000.csv',dtype=None,delimiter=",")

# store voltage values in rawData array
rawData=data[:,1]
rawData=rawData-np.mean(rawData)


# store time values
timeArr = data[:,0]

# take fft of raw data
yf=fft(rawData)

# number of samples in data
N=30000

# construct x axis for 1-sided (positive freq) FFT plot
# Sampling rate is 2.5 GS/s
xf = np.linspace(0.0, 1.0/2.0*2.5*10**9, N//2)

# plot fft
plt.figure()
plt.plot(xf,2.0/N * np.abs(yf[0:N//2]))


# filter carrier envelope
yf[0:1300] = yf[1300:2600]
yf[-1300:]  = yf[-2600:-1300]


# plot filtered spectrum
plt.figure()
plt.plot(xf,2.0/N * np.abs(yf[0:N//2]))


# take inverse fft of filtered spectrum
signal = ifft(yf)


# plot signal
plt.figure()
plt.plot(timeArr,signal)


# truncate signal
signalTruncated = signal[14800:15550]


# plot truncated signal
plt.figure()
plt.plot(timeArr[14800:15550],signalTruncated)


# take hilbert transform of truncated signal
z= hilbert(signalTruncated.real) #form the analytical signal
inst_amplitude = np.abs(z) #envelope extraction
inst_phase = np.unwrap(np.angle(z))#inst phase
inst_freq = np.diff(inst_phase)/(2*np.pi)*2.5*10**9 #inst frequency


# plot truncated signal with carrier envelope extracted
plt.figure()
plt.plot(timeArr[14800:15550],signalTruncated)
plt.plot(timeArr[14800:15550],inst_amplitude)


#plot instantaneuos frequency
plt.figure()
plt.plot(timeArr[14800:15549],inst_freq)


#filter signal to extract carrier envelope
yf=fft(rawData)
yf[1300:-1300]=0
carrier=ifft(yf)
carrier_trunc = carrier[14800:15550]


# smooth inst_freq
df = pd.DataFrame(inst_freq)
inst_freq_ave = df.rolling(15).mean()


# plot carrier and inst_freq
plt.figure()
ax1 = plt.subplot(211)
plt.plot(timeArr[14800:15549]*10**9,carrier_trunc.real[0:-1],c='black')
plt.ylabel('Amplitude (arb. units)',size=13)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.title('Chirp Measurement',size=15)
plt.xticks(size=12.5)
plt.yticks(size=12.5)
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(timeArr[14800:15549]*10**9,inst_freq_ave*10**-6-350,c='black')
plt.ylabel('Frequency chirp (MHz)',size=13)
plt.xlabel('Time (nsec)',size=13)
plt.xticks(size=12.5)
plt.yticks(size=12.5)
plt.subplots_adjust(wspace=0, hspace=0)
ax1.tick_params(which='both',direction='in',top=True,right=True)
ax2.tick_params(which='both',direction='in',top=True,right=True)
plt.minorticks_on()
plt.setp(ax1.spines.values(), linewidth=1)
plt.setp(ax2.spines.values(), linewidth=1)

# calculate average frequency offset
carrier_integral=np.trapz(carrier_trunc.real,timeArr[14800:15550]*10**9)

integrandA = np.array(1/carrier_integral*(inst_freq_ave*10**-6-350))
integrandA=integrandA.flatten()

integrand = integrandA*carrier_trunc.real[0:-1]

ave_offset = np.trapz(integrand[15:],timeArr[14815:15549]*10**9)


print(ave_offset)








