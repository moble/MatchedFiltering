#! /usr/bin/python

import sys
import numpy as np
import scipy.interpolate as spi
import pyMatchedFiltering as MatchedFiltering
from numpy.random import rand as random
from numpy.random import randint as randomint

Spins = ['Spinning', 'Nonspinning']
Msun = 4.92579497e-6 # seconds
dt = 1.0/44100.0
#dt = 1.0/45000.0
#dt = 6.103515625e-05

CutoffFrequency = 5.0 # Hz

Data = [None]*2
Mag = [None]*2
Arg = [None]*2
MaxMagIndex = [None]*2
InitialFrequency = [None]*2

for i,File in enumerate(["rhOverM_EqualMassAlignedSpins_L2_M2.dat", "rhOverM_EqualMassNonspinning_L2_M2.dat"]) :
    print("Reading {}...  ".format(File))
    Data[i] = np.genfromtxt('Sounds/'+File)
    Mag[i] = spi.interp1d(Data[i][:,0], Data[i][:,1], bounds_error=False, fill_value=0.0)
    Arg[i] = spi.interp1d(Data[i][:,0], Data[i][:,2], bounds_error=False, fill_value=0.0)
    MaxMagIndex[i] = np.argmax(Data[i][:,1])
    InitialFrequency[i] = -(np.diff(Data[i][0:2,2])/(2*np.pi*Msun*np.diff(Data[i][0:2,0])))[0]

for i in range(1,21) :
    M = 5.0*random() + 5.0
    S = randomint(2)
    Offset = randomint(Mag[S].len)
    print("nws{0:02d}.wav\tM={0}MSun\t{1}...".format(i,M,Spins[S]))
    initialFrequency = InitialFrequency[S]/float(M)
    if(initialFrequency<CutoffFrequency) :
        print("\t\tinitialFrequency={}<20Hz.  Finding appropriate initial time...".format(initialFrequency))
        TofF = spi.interp1d(-np.diff(Data[S][:MaxMagIndex[S],2])/(2*np.pi*M*Msun*np.diff(Data[S][:MaxMagIndex[S],0])),
                            M*Msun*Data[1:MaxMagIndex[S],0])
        t0 = TofF(20)
    else :
        t0 = Data[S][0,0]*M*Msun
    t1 = Data[S][-1,0]*M*Msun
    tOverM = np.arange(t0/(M*Msun), t1/(M*Msun) + 0.05*(t1-t0)/(M*Msun), dt/(M*Msun))
    W = MatchedFiltering.Waveform()
    W.dt = dt
    W.N = len(tOverM)
    W.data = Mag[S](tOverM)*np.sin(Arg[S](tOverM)-Arg[S](tOverM[0]))
    W.WriteWAVFile('SearchData/nws{0:02d}.wav'.format(i))
print("Finished")
