#! /usr/bin/python

import sys
import numpy as np
import scipy.interpolate as spi
import scipy.signal
import pyMatchedFiltering as MatchedFiltering
from numpy.random import rand as random
from numpy.random import randint as randomint

Spins = ['Spinning', 'Nonspinning']
Msun = 4.92579497e-6 # seconds
dt = 1.0/44100.0
#dt = 1.0/45000.0
#dt = 6.103515625e-05
t_tot = 20.0
t_pad = 0.5
# t_window = 0.5
# N_window = int(t_window/dt)
# window = np.blackman(N_window)

# Now read in the LIGO noise data and adjust to suit this waveform
LIGONoise = MatchedFiltering.Waveform('Sounds/LIGONoise.wav')
LIGONoise.data = scipy.signal.resample(LIGONoise.data, LIGONoise.N*LIGONoise.dt/dt)
LIGONoise.N = len(LIGONoise.data)
LIGONoise.dt = dt

CutoffFrequency = 5.0 # Hz

Data = [None]*2
Mag = [None]*2
Arg = [None]*2
MaxMagIndex = [None]*2
InitialFrequency = [None]*2

for i,File in enumerate(["rhOverM_EqualMassAlignedSpins_L2_M2.dat", "rhOverM_EqualMassNonspinning_L2_M2.dat"]) :
    # print("Reading {}...  ".format(File))
    Data[i] = np.genfromtxt('Sounds/'+File)
    Mag[i] = spi.interp1d(Data[i][:,0], Data[i][:,1], bounds_error=False, fill_value=0.0)
    Arg[i] = spi.interp1d(Data[i][:,0], Data[i][:,2], bounds_error=False, fill_value=0.0)
    MaxMagIndex[i] = np.argmax(Data[i][:,1])
    InitialFrequency[i] = -(np.diff(Data[i][0:2,2])/(2*np.pi*Msun*np.diff(Data[i][0:2,0])))[0]

for i in range(1, 23) :
    # Randomly select some parameters
    M = 310.0*random() + 10.0
    S = randomint(2)
    # Create the waveform appropriately
    # t0 = Data[S][0,0]*M*Msun
    t1 = Data[S][-1,0]*M*Msun
    tOverM = np.arange((t1-t_tot)/(M*Msun), t1/(M*Msun) + t_pad/(M*Msun), dt/(M*Msun))
    if(len(tOverM)>LIGONoise.N) :
        tOverM = tOverM[len(tOverM)-LIGONoise.N-1:]
    W = MatchedFiltering.Waveform()
    W.dt = dt
    W.N = len(tOverM)
    W.data = Mag[S](tOverM)*np.sin(Arg[S](tOverM)-Arg[S](tOverM[0]))
    # W.data[:N_window/2] = W.data[:N_window/2]*window[:N_window/2]
    if(W.N>LIGONoise.N) :
        W.data = W.data[W.N-LIGONoise.N:]
        W.N = LIGONoise.N
    else :
        W.PadWithZeroToSameLengthAs(LIGONoise)
        print "\tLengthening data"
    Offset = randomint(W.N)
    W.data = np.roll(W.data, Offset)
    # Combine the signals
    SignalToNoiseRatio = 1.e-2 # Use a large number because the templates will be widely separated
    FakeLIGOData = (1.0-SignalToNoiseRatio)*LIGONoise + SignalToNoiseRatio*W
    # Write the fake LIGO data to file
    FakeLIGOData.WriteWAVFile('SearchData/nws{0:02d}.wav'.format(i))
    print("nws{0:02d}.wav\tM={1}MSun\t{2}\tOffset={3}/{4}={5}...".format(i,M,Spins[S],Offset,W.data.size,W.Time()[Offset]))
