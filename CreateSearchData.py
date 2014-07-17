#! /usr/bin/env python

"""Create data for use in MatchedFiltering demonstration

This script creates a series of somewhat random data files for use
with the main MatchedFiltering notebook.  At the end of the notebook,
the student is challenged to figure out the parameters for their
assigned data set.  Each computer gets a different data file.

"""

import os
import sys
import random
import numpy
import Utils

import os
if not os.path.exists('SearchData'):
    os.makedirs('SearchData')

random.seed(1234)

SamplingFrequency = Utils.DefaultSamplingFrequency

DistanceInMegaparsecs = 10.0 # Note that this controls the signal-to-noise ratio

def NonspinningGW(MassInSolarMasses, DistanceInMegaparsecs=DistanceInMegaparsecs):
    return Utils.Waveform('Data/rhOverM_EqualMassNonspinning_L2_M2.dat',
                          SamplingFrequency, MassInSolarMasses, DistanceInMegaparsecs)
def SpinningGW(MassInSolarMasses, DistanceInMegaparsecs=DistanceInMegaparsecs):
    return Utils.Waveform('Data/rhOverM_EqualMassAlignedSpins_L2_M2.dat',
                          SamplingFrequency, MassInSolarMasses, DistanceInMegaparsecs)

GWs = [NonspinningGW, SpinningGW] # Two possibilities for the spins
Masses = numpy.arange(5., 105., 5.) # Choose integer masses to make things easier

LIGONoise = Utils.Waveform('Data/LIGONoise.dat')

for i in range(1, 23) :
    # Randomly select some parameters
    GW = random.choice(GWs)
    Mass = random.choice(Masses)
    dt = random.uniform(0., Utils.DefaultWaveformLength)
    # Evaluate the waveform
    W = GW(Mass).Roll(dt)
    # Combine the signals to create fake LIGO data
    W.data += LIGONoise.data
    # Write the fake LIGO data to file
    numpy.savetxt('SearchData/nws{0:02d}.dat'.format(i),
                  W.data,
                  header='dt = {0}'.format(W.dt))
    print("nws{0:02d}.dat\tMass={1}MSun\t{2}\tOffset={3}".format(i,Mass,GW.__name__,dt))
