#! /usr/bin/python

import sys
import numpy as np
import scipy.interpolate as spi
import MatchedFiltering

Msun = 4.92579497e-6 # seconds
dt = 1.0/44100.0
#dt = 1.0/45000.0
#dt = 6.103515625e-05

CutoffFrequency = 5.0 # Hz

for File in ["rhOverM_EqualMassAlignedSpins_L2_M2.dat", "rhOverM_EqualMassNonspinning_L2_M2.dat"] :
    print("Working on {}...  ".format(File))
    Data = np.genfromtxt('Sounds/'+File)
    Mag = spi.interp1d(Data[:,0], Data[:,1], bounds_error=False, fill_value=0.0)
    Arg = spi.interp1d(Data[:,0], Data[:,2], bounds_error=False, fill_value=0.0)
    MaxMagIndex = np.argmax(Data[:,1])
    InitialFrequency = -(np.diff(Data[0:2,2])/(2*np.pi*Msun*np.diff(Data[0:2,0])))[0]
    for M in [5, 6, 10, 20, 40, 80, 160, 320] :
        print("\tM={0}MSun...".format(M))
        initialFrequency = InitialFrequency/float(M)
        if(initialFrequency<CutoffFrequency) :
            print("\t\tinitialFrequency={}<20Hz.  Finding appropriate initial time...".format(initialFrequency))
            TofF = spi.interp1d(-np.diff(Data[:MaxMagIndex,2])/(2*np.pi*M*Msun*np.diff(Data[:MaxMagIndex,0])),
                                M*Msun*Data[1:MaxMagIndex,0])
            t0 = TofF(20)
        else :
            t0 = Data[0,0]*M*Msun
        t1 = Data[-1,0]*M*Msun
        tOverM = np.arange(t0/(M*Msun), t1/(M*Msun) + 0.05*(t1-t0)/(M*Msun), dt/(M*Msun))
        W = MatchedFiltering.Waveform()
        W.dt = dt
        W.N = len(tOverM)
        W.data = Mag(tOverM)*np.sin(Arg(tOverM)-Arg(tOverM[0]))
        W.WriteWAVFile('Sounds/'+File.replace("rhOverM_","").replace("L2_M2.dat","")+'Re_M{0:03d}.wav'.format(M))
    print("")
print("Finished")
