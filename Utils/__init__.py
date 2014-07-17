import sys, os
import numpy
import scipy
import scipy.interpolate
import pyaudio

sys.path.append(os.path.dirname(__file__))

Msun = 4.92579497e-6 # seconds
SecondsPerMegaparsec = 1.02927125e14 # seconds
DefaultSamplingFrequency = 8192 # Hertz [integer]
DefaultWaveformLength = 20.0 # seconds
PaddingLength = 0.5 # seconds [extra time for ringdown]
WindowLength = 0.5 # seconds [amount of time needed to turn on]

def _WindowFunction(NormalizedTime):
    """Window data, turning on from 0.0 to 1.0"""
    w = numpy.empty_like(NormalizedTime)
    for i,t in enumerate(NormalizedTime):
        if t<=1.0e-15:
            w[i] = 0.0
        elif t-1.0>=-1.0e-15:
            w[i] = 1.0
        else:
            exponent = 1.0/t - 1.0/(1.0-t)
            if exponent>450.0:
                w[i] = 0.0
            else:
                w[i] = 1.0 / (1.0 + numpy.exp(exponent))
    return w
    # return numpy.array([0.0 if t<=1.0e-15 else (1.0 if t-1.0>=-1.0e-15 else 1.0 / (1.0 + numpy.exp(1.0/t - 1.0/(1-t)))) for t in NormalizedTime])

class Waveform :
    """Container for time-domain data.

    An object of class Waveform contains the time-step size 'dt', the
    total number of time steps 'N', and real-valued data for each of
    those time steps 'data'.

    You can initialize a Waveform by giving the name of some data file
    or by recording through the computer's speakers.  For example:
    >>> import MatchedFiltering.Utils
    >>> SamplingFrequency = 8192
    >>> TotalMassInSolarMasses = 10.0
    >>> DistanceInMegaparsecs = 100.0
    >>> s = MatchedFiltering.Waveform('rhOverM_L2_M2.dat', SamplingFrequency, TotalMassInSolarMasses, DistanceInMegaparsecs)
    >>> Noise = MatchedFiltering.RecordWaveform(10)

    Waveforms can also be added to each other and multiplied by a
    number to make them louder or quieter.

    """

    def __init__(self, *args) :
        if(len(args) == 0) :
            self.dt = 0.0
            self.N = 0
            self.data = numpy.empty(0)
        elif(len(args) == 1) :
            if(isinstance(args[0], Waveform)) : # If the input argument is another Waveform, just copy it
                self.dt = args[0].dt
                self.N = args[0].N
                self.data = numpy.array(args[0].data)
            else : # Otherwise, assume it's a file and try to read it
                with open(args[0], 'r') as f:
                    first_line = f.readline()
                self.dt = float(first_line[7:-1])
                self.data = numpy.loadtxt(args[0])
                self.N = len(self.data)
        elif(len(args) == 4) :
            FileName = args[0]
            SamplingFrequency = args[1]
            TotalMassInSolarMasses = args[2]
            DistanceInMegaparsecs = args[3]
            filedata = numpy.loadtxt(FileName)
            tIn = filedata[:,0] * TotalMassInSolarMasses * Msun
            Mag = scipy.interpolate.splrep(tIn, filedata[:,1] * TotalMassInSolarMasses * Msun / (DistanceInMegaparsecs*SecondsPerMegaparsec), k=1, s=0)
            Arg = scipy.interpolate.splrep(tIn, filedata[:,2], k=1, s=0)
            tLast = tIn[-1] + PaddingLength
            self.dt = 1.0/SamplingFrequency
            Time = numpy.linspace(tLast - DefaultWaveformLength, tLast, int(DefaultWaveformLength/self.dt), endpoint=True)
            self.N = len(Time)
            self.data = scipy.interpolate.splev(Time, Mag) * numpy.sin(scipy.interpolate.splev(Time, Arg)-scipy.interpolate.splev(Time[0], Arg))
            # Roll the data on slowly at the beginning
            NWindow = int(WindowLength/self.dt)
            self.data[:NWindow] = self.data[:NWindow]*_WindowFunction((Time[:NWindow]-Time[0])/(Time[NWindow]-Time[0]))
            # Roll off the data at the end by extrapolating the phase and logarithmic amplitude naively, and blending that into the real data
            N_pad = int(PaddingLength/self.dt)
            t1 = tIn[-1]
            tm = tIn[numpy.argmax(numpy.abs(filedata[:,1]))]
            t0 = 0.5*(t1+tm)
            i1 = numpy.argmin(numpy.abs(Time-t1))
            i0 = numpy.argmin(numpy.abs(Time-t0))
            chi0 = numpy.log(scipy.interpolate.splev(t0, Mag))
            chi1 = numpy.log(scipy.interpolate.splev(t1, Mag))
            phi0 = scipy.interpolate.splev(t0, Arg)-scipy.interpolate.splev(Time[0], Arg)
            phi1 = scipy.interpolate.splev(t1, Arg)-scipy.interpolate.splev(Time[0], Arg)
            w = _WindowFunction((t1-Time[i0:i1])/(t1-t0))
            self.data[i0:i1] = w * self.data[i0:i1] \
                               + (1.0-w) * numpy.exp(((chi1-chi0)/(t1-t0))*(Time[i0:i1]-t0)+chi0) * numpy.sin(((phi1-phi0)/(t1-t0))*(Time[i0:i1]-t0)+phi0)
            self.data[i1:] = numpy.exp(((chi1-chi0)/(t1-t0))*(Time[i1:]-t0)+chi0) * numpy.sin(((phi1-phi0)/(t1-t0))*(Time[i1:]-t0)+phi0)
        else :
            raise ValueError('Unrecognized number of arguments to Waveform constructor')

    def fft(self) :
        return self.dt * numpy.fft.fft(self.data)

    @property
    def t(self) :
        """
        Return array of the time data.

        This function returns all the values of time at which the data
        is known.
        """
        return numpy.linspace(0.0, (self.N-1)*self.dt, self.N, endpoint=True)

    @property
    def f(self) :
        """Array of single-sided frequency data"""
        return numpy.fft.fftfreq(self.N, self.dt)

    def PadWithZeroToSameLengthAs(self, other) :
        self.data = numpy.append(self.data, [0.0]*(other.N-self.N))
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
        self.data = numpy.interp(t, self.Time(), self.data, 0.0, 0.0)
        self.dt = t[1]-t[0]
        self.N = len(t)
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
        if( (self.N==other.N) and (self.dt==other.dt) ) :
            a = Waveform(self)
            a.data += other.data
            return a
        a = Waveform(self)
        b = Waveform(other)
        t1 = max((a.N-1)*a.dt, (b.N-1)*b.dt)
        dt = min(a.dt, b.dt)
        t = numpy.linspace(0.0, t1, int(t1/dt)+1) # Make sure that t1 is included
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

    def Roll(self, TimeOffset) :
        """
        Return a new Waveform with time shifted periodically.

        Because of the periodic nature of finite signals assumed by
        the Fourier transform, we can shift a signal in time by simply
        moving some of the data from the end of the array to the
        beginning.
        """
        a = Waveform(self)
        a.data = numpy.roll(self.data, int(TimeOffset/self.dt))
        return a

    def Play(self) :
        """Play the Waveform as a sound through the computer's speakers.

        Given a Waveform object 'W', you can play it with the command
        >>> W.Play()

        Note that the volume is adjusted so that the loudest part of
        the Waveform is precisely as loud as possible (no more; no
        less).  But then, the signal passes through the sound card and
        speaker, which may adjust the volume themselves.

        """
        from IPython.display import Audio, display
        display(Audio(self.data, rate=1.0/self.dt, autoplay=True))

    def Audio(self) :
        """Play the Waveform as a sound through the computer's speakers.

        Given a Waveform object 'W', you can display its audio widget
        it with the command
        >>> W.Audio()
        Then click on the play button to listen to it.

        Note that the volume is adjusted so that the loudest part of
        the Waveform is precisely as loud as possible (no more; no
        less).  But then, the signal passes through the sound card and
        speaker, which may adjust the volume themselves.

        """
        from IPython.display import Audio, display
        display(Audio(self.data, rate=1.0/self.dt, autoplay=False))


def _InnerProduct(W1FFT, W2FFT, PSDList, df) :
    """
    Evaluate inner product of two frequency-domain signals.

    This is a very simple function that assumes the input data all
    have the correct size, and are of the correct type.
    """
    return 4*df*sum(W1FFT * numpy.conj(W2FFT) / PSDList).real


def _TimeToPositiveFrequencies(N, dt) :
    """
    Return the single-sided frequency-space equivalent.
    """
    # if(N&(N-1)) : ## Test if N is a power of 2
    #     raise ValueError("len(Time)={} is not a power of 2.".format(N))
    n = 1 + (N/2)
    df = 1.0 / (N*dt)
    return numpy.arange(n) * df


def Match(W1, W2, Noise) :
    import numpy.fft as npfft
    if((not isinstance(W1, Waveform)) or (not isinstance(W2, Waveform)) or (not isinstance(Noise, Waveform))) :
        ErrorString = \
        """You gave me "{0}", "{1}", and "{2}" objects.  I need "Waveform"
        objects.  Try again.""".format(W1.__class__.__name__, W2.__class__.__name__,
                                       Noise.__class__.__name__)
        raise TypeError(ErrorString)
    if((not W1.N==W2.N) or (not W1.dt==W2.dt) or (not W1.N==Noise.N) or (not W1.dt==Noise.dt)) :
        raise ValueError("Disagreement among input sizes")
    N = W1.N
    dt = W1.dt
    df = 1.0/(N*dt)
    psd = abs(Noise.fft())**2
    SeismicWall = 35.0 # Hz
    a = W1.fft()
    b = W2.fft()
    a /= W1.dt*numpy.sqrt(_InnerProduct(a, a, psd, df))
    b /= W2.dt*numpy.sqrt(_InnerProduct(b, b, psd, df))
    integrand = a * numpy.conj(b) / psd
    integrand[:int(SeismicWall/df)] = 0.0 # Zero below the seismic wall
    integrand[-int(SeismicWall/df):] = 0.0 # Zero above the negative seismic wall
    integrand[len(integrand)/4:3*len(integrand)/4] = 0.0 # Zero above/below a simple upper frequency limit
    return 4*W1.dt*abs(npfft.ifft( integrand ))

    # a = dt*npfft.rfft(a.data)
    # b = dt*npfft.rfft(b.data)
    # a /= dt * numpy.sqrt(_InnerProduct(a, a, psd, df))
    # b /= dt * numpy.sqrt(_InnerProduct(b, b, psd, df))
    # return 2*dt*abs(npfft.irfft( a * numpy.conj(b) / psd ))


def SavitzkyGolay(y, window_size, order, deriv=0):
    r"""
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    CREDIT: http://www.scipy.org/Cookbook/SavitzkyGolay

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = numpy.linspace(-4, 4, 500)
    y = numpy.exp( -t**2 ) + numpy.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, numpy.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = numpy.abs(numpy.int(window_size))
        order = numpy.abs(numpy.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = numpy.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = numpy.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - numpy.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + numpy.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = numpy.concatenate((firstvals, y, lastvals))
    return numpy.convolve( m, y, mode='valid')
