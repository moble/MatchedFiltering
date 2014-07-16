import sys, os
import numpy as np
import scipy as sp
import SmoothData
import pyaudio

sys.path.append(os.path.dirname(__file__))

DefaultSamplingFrequency = int(pyaudio.PyAudio().get_device_info_by_index(0)['defaultSampleRate'])

class Waveform :
    """
    Container for time-domain data.

    An object of class Waveform contains an initial time 't0', a
    time-step size 'dt', the total number of time steps 'N', and
    real-valued data for each of those time steps 'data'.

    You can initialize a Waveform by giving the name of some WAV file
    or by recording through the computer's speakers.  For example:
    >>> import MatchedFiltering
    >>> W1 = MatchedFiltering.Waveform('test.wav')
    >>> W2 = MatchedFiltering.RecordWaveform(10)

    Waveforms can also be added to each other and multiplied by a
    number to make them louder or quieter.
    """

    def __init__(self, *args) :
        if(len(args) == 0) :
            self.t0 = 0.0
            self.dt = 0.0
            self.N = 0
            self.data = np.empty(0)
        elif(len(args) == 1) :
            if(isinstance(args[0], Waveform)) : # If the input argument is another Waveform, just copy it
                self.t0 = args[0].t0
                self.dt = args[0].dt
                self.N = args[0].N
                self.data = np.array(args[0].data)
            else : # Otherwise, assume it's a file and try to read it
                import ReadWAVFile
                self.data, SampleRate, self.N = ReadWAVFile.ReadWAVFile(args[0])
                self.dt = 1.0/float(SampleRate)
                self.t0 = 0.0
        else :
            raise ValueError('Unrecognized number of arguments to Waveform constructor')

    def fft(self) :
        return self.dt * np.fft.fft()

    def Time(self) :
        """
        Return array of the time data.

        This function returns all the values of time at which the data
        is known.
        """
        return np.linspace(self.t0, self.t0+(self.N-1)*self.dt, self.N, endpoint=True)

    def PadWithZeroToSameLengthAs(self, other) :
        self.data = np.append(self.data, [0.0]*(other.N-self.N))
        self.N = other.N
        return self

    def Interpolate(self, t) :
        """
        Interpolate the Waveform onto the given time data.

        If the argument is another Waveform, this Waveform is just
        interpolated onto the time data from that one.

        Note that if the given times are outside the times known to
        this Waveform, the "interpolated" data is set to 0.
        """
        if(isinstance(t, Waveform)) :
            return self.InterpolateTime(t.Time())
        self.data = np.interp(t, self.Time(), self.data, 0.0, 0.0)
        self.t0 = t[0]
        self.dt = t[1]-t[0]
        self.N = len(t)
        return self

    def AddToTime(self, t0) :
        """
        Shift the Waveform's time axis by adding 't0'.
        """
        self.t0 = t0
        return self

    def __add__(self, other) :
        """
        Add data from the second Waveform to the first.

        This function returns a new Waveform with the combined data
        from two separate Waveform objects.  It may be called with
        something like
        >>> W3 = W1 + W2

        Note that the Waveforms may have different lengths, in which
        case the returned Waveform just has the longer length; the
        shorter Waveform is assumed to just be 0 when it does not
        exist.  Also, the Waveforms may have different time steps
        'dt', in which case the returned Waveform has the finer time
        step.  However, naive interpolation is done in this case
        (rather than upsampling), which might cause subtle problems.
        """
        if( (self.t0==other.t0) and (self.N==other.N) and (self.dt==other.dt) ) :
            a = Waveform(self)
            a.data += other.data
            return a
        a = Waveform(self)
        b = Waveform(other)
        t0 = min(a.t0, b.t0)
        t1 = max(a.t0+(a.N-1)*a.dt, b.t0+(b.N-1)*b.dt)
        dt = min(a.dt, b.dt)
        t = np.linspace(t0, t1, int((t1-t0)/dt)+1) # Make sure that t0 and t1 are included
        a.Interpolate(t)
        b.Interpolate(t)
        for i in range(len(a.data)) :
            a.data[i] += b.data[i]
        return a

    def __mul__(self, scale) :
        """
        Return a new Waveform with data multiplied by 'scale'.
        """
        a = Waveform(self)
        a.data *= scale
        return a

    def __rmul__(self, scale) :
        """
        Return a new Waveform with data multiplied by 'scale'.
        """
        return self*scale

    def Rotate(self, time) :
        """
        Return a new Waveform with time shifted periodically.

        Because of the periodic nature of finite signals assumed by
        the Fourier transform, we can shift a signal in time by simply
        moving some of the data from the end of the array to the
        beginning.
        """
        a = Waveform(self)
        i = int(time/a.dt)
        a.data = np.concatenate((a.data[-i:], a.data[:-i]))
        return a

    def Play(self) :
        """
        Play the Waveform as a sound through the computer's speakers.

        Given a Waveform object 'W', you can play it with the command
        >>> W.Play()

        Note that the volume is adjusted so that the loudest part of
        the Waveform is precisely as loud as possible (no more; no
        less).  But then, the signal passes through the sound card and
        speaker, which may adjust the volume themselves.
        """
        import wave
        import struct
        chunk = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format = pyaudio.paInt32,
                        channels = 1,
                        rate = int(1.0/self.dt),
                        output = True)
        scale = min(2147483647.0/max(self.data), -2147483648.0/min(self.data))
        data = scale*self.data
        data = struct.pack('%ii'%(len(self.data)), *data)
        stream.write(data)
        stream.close()
        p.terminate()

    def WriteWAVFile(self, WAVFileName) :
        """
        Write the Waveform to file as a WAV audio file.

        With a Waveform object 'W', you can call this function with
        something along the lines of:
        >>> W.WriteWAVFile('SomeFileName.wav')
        """
        import wave
        import struct
        chunk = 1024
        stream = wave.open(WAVFileName, "wb")
        # stream = p.open(format = pyaudio.paInt32,
        #                 channels = 1,
        #                 rate = int(1.0/self.dt),
        #                 output = True)
        scale = min(2147483647.0/max(self.data), -2147483648.0/min(self.data))
        data = scale*self.data
        data = struct.pack('%ii'%(len(self.data)), *data)
        stream.setnchannels(1)
        stream.setframerate(int(1.0/self.dt))
        stream.setsampwidth(4)
        stream.setnframes(self.N)
        stream.writeframes( data )
        stream.close()



def RecordWaveform(RecordingTime=5.0, SamplingFrequency=DefaultSamplingFrequency, Normalized=True) :
    """
    Record audio from the computer's microphone.

    A Waveform object is returned, containing the recorded data.  The
    optional arguments are the number of seconds for which to record,
    the sampling frequency (in Hertz), and whether or not the maximum
    absolute value in the data is 1.
    """
    import wave
    import struct

    chunk = 1024
    FORMAT = pyaudio.paInt32 # Record as signed "4-byte" ints
    fmt = "%ii" # read signed "4-byte" ints
    offset = 0
    scale = 2147483648.0
    CHANNELS = 1

    ## Do the actual recording
    print("Recording..."); sys.stdout.flush()
    p = pyaudio.PyAudio()
    all = range(0, SamplingFrequency / chunk * RecordingTime)
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = SamplingFrequency,
                    input = True,
                    frames_per_buffer = chunk)
    for i in range(0, SamplingFrequency / chunk * RecordingTime) :
        data = stream.read(chunk)
        all[i] = data
    stream.close()
    p.terminate()
    print("Done recording.")

    ## The data are stored as a string of "4-byte" integers in the
    ## range [-2147483648,2147483648).  We need to convert this to
    ## floats in the range [-1,1), and then create the Waveform
    ## object.
    data = ''.join(all)
    W = Waveform()
    W.t0 = 0.0
    W.dt = 1.0/float(SamplingFrequency)
    W.N = len(data)/struct.calcsize('1i')
    fmt = fmt % W.N
    W.data = (np.array(struct.unpack(fmt, data), dtype=float)-offset) / scale
    if(Normalized) :
        W.data = W.data / max(abs(W.data))
    return W


def _InnerProduct(W1FFT, W2FFT, PSDList, df) :
    """
    Evaluate inner product of two frequency-domain signals.

    This is a very simple function that assumes the input data all
    have the correct size, and are of the correct type.
    """
    return 4*df*sum(W1FFT * np.conj(W2FFT) / PSDList).real

def _TimeToPositiveFrequencies(N, dt) :
    """
    Return the single-sided frequency-space equivalent.
    """
    # if(N&(N-1)) : ## Test if N is a power of 2
    #     raise ValueError("len(Time)={} is not a power of 2.".format(N))
    n = 1 + (N/2)
    df = 1.0 / (N*dt)
    return np.arange(n) * df

def Match(W1, W2, Noise) :
    """
    Return match as function of time offset between two Waveforms.


    """
    import numpy as np
    import scipy.interpolate as spi
    import numpy.fft as npfft
    import Utils
    Waveform = Utils.Waveform
    if((not isinstance(W1, Waveform)) or (not isinstance(W2, Waveform))) :
        ErrorString = \
        """You gave me "{0}", "{1}", and "{2}" objects.  I need "Waveform"
        objects.  Try again.""".format(W1.__class__.__name__, W2.__class__.__name__,
                                       Noise.__class__.__name__)
        raise TypeError(ErrorString)
    a = W1
    b = W2
    c = Noise
    if((not a.N==b.N) or (not a.dt==b.dt) or (not a.N==c.N) or (not a.dt==c.dt)) :
        a = Waveform(W1)
        b = Waveform(W2)
        c = Waveform(Noise)
        t0 = min(a.t0, b.t0, c.t0)
        #print(a.t0, b.t0, c.t0, t0)
        t1 = max(a.t0+(a.N-1)*a.dt, b.t0+(b.N-1)*b.dt, c.t0+(c.N-1)*c.dt)
        #print(a.t0+(a.N-1)*a.dt, b.t0+(b.N-1)*b.dt, c.t0+(c.N-1)*c.dt, t1)
        dt = min(a.dt, b.dt, c.dt)
        #print(a.dt, b.dt, c.dt, dt)
        t = np.linspace(t0, t1, int((t1-t0)/dt)+1) # Make sure that t0 and t1 are included
        #print(len(t))
        a.Interpolate(t)
        b.Interpolate(t)
        c.Interpolate(t)
    N = a.N
    dt = a.dt
    df = 1.0/(N*dt)
    Frequencies = npfft.fftfreq(Noise.N, Noise.dt)
    Frequencies = np.concatenate((Frequencies[len(Frequencies)/2:], Frequencies[:len(Frequencies)/2]))
    # PSD = Noise.dt * npfft.fft(Noise.data)
    PSD = npfft.fft(Noise.data) / Noise.N
    PSD = abs(PSD)**2
    PSD = np.concatenate((PSD[len(PSD)/2:], PSD[:len(PSD)/2]))
    PSD = PSD + PSD[::-1] # Make sure it's symmetric
    SmoothedPSD = SmoothData.SavitzkyGolay(PSD, 101, 2)
    # SmoothedPSD = SmoothData.SavitzkyGolay(PSD, 3, 0)
    PSD = spi.UnivariateSpline(Frequencies, SmoothedPSD, s=0)
    f = _TimeToPositiveFrequencies(N, dt)
    psd = PSD(f)
    a = dt*npfft.rfft(a.data)
    b = dt*npfft.rfft(b.data)
    a /= dt * np.sqrt(_InnerProduct(a, a, psd, df))
    b /= dt * np.sqrt(_InnerProduct(b, b, psd, df))
    # return [2*dt*abs(npfft.irfft( a * np.conj(b) / psd )), dt, N]
    return 2*dt*abs(npfft.irfft( a * np.conj(b) / psd ))

# def Match(W1, W2, PSD) :
#     """
#     Return maximum overlap between the two Waveforms.
#     """
#     return max(Overlap(W1, W2, PSD))


def SimplifyNoisyData(x, y, n=10000) :
    """
    Reduce number of points in (x,y) by splitting into bins and finding max and min of each.

    Without some simplification, matplotlib fails when asked to plot
    large data sets.  But if we just plot every tenth point (say) of
    noisy data, the peaks look different.  It would be better to go
    through the data to find the max and min values in every ten
    points, and then plot those.  That is the basic idea behind this
    function.

    Note that `n` is the number of bins for the max and for the min; the returned data has 2*n data points in X and Y.

    Based on <http://stackoverflow.com/a/8881973/1194883>

    """

    if(y.size%(2*n)==0) :
        N = n
    else :
        N = n-1

    # Divide into chunks
    ychunksize = y.size // N
    ychunks = y[:ychunksize*N].reshape((-1, ychunksize))
    xchunksize = y.size // (2*N)
    xchunks = x[:xchunksize*2*N].reshape((-1, xchunksize))

    # Calculate the max and min of chunksize-element chunks...
    max_env = ychunks.max(axis=1)
    min_env = ychunks.min(axis=1)
    X = xchunks.mean(axis=1)

    # If necessary, include the missing end
    if(n!=N) :
        max_env = np.append(max_env, np.max(y[ychunksize*N:]))
        min_env = np.append(min_env, np.min(y[ychunksize*N:]))
        X = np.append(X, [(X[-1]+x[-1])/2., x[-1]])

    # Interleave the max and min to form Y
    Y = np.ravel([max_env, min_env], order='F')

    return X,Y
