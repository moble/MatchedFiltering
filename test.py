cd ~/Research/Talks/REU2012/
import MatchedFiltering
import MatchedFiltering.SmoothData
import scipy.interpolate

## First, record the baseline noise level, and FFT it.
Noise = MatchedFiltering.RecordWaveform(10)
NoiseFFT = Noise.dt * np.fft.fft(Noise.data)

## These are the frequencies at which the NoiseFFT is known:
f_Noise = np.fft.fftfreq(Noise.N, Noise.dt)
f_Noise = np.concatenate((f_Noise[len(f_Noise)/2+1:], f_Noise[:len(f_Noise)/2+1]))

## Next, convert this to a PSD by squaring the absolute value and
## dividing by the number of points.
PSDData = abs(NoiseFFT)**2
PSDData = np.concatenate((PSDData[len(PSDData)/2+1:], PSDData[:len(PSDData)/2+1]))

## Now, smooth the PSD for easier interpolation.
SmoothedPSD = MatchedFiltering.SmoothData.SavitzkyGolay(PSDData, 4001, 2)

## Construct the interpolation function.
PSD = scipy.interpolate.UnivariateSpline(f_Noise, SmoothedPSD, s=0)

## Plot the smoothed PSD.
semilogy(f_Noise, SmoothedPSD)

## Create a sine-gaussian to serve as our signal
SineGaussianCenter = 5.0 ## Center of the function in seconds
SineGaussianFWHM = 1.9 ## FWHM of the gaussian in seconds
SineGaussianFreq = 80 ## Frequency of the sine in Hz
SineSweep = SineGaussianFreq/4.0
sigma = SineGaussianFWHM / (2*sqrt(2*log(2)))
SineGaussian = MatchedFiltering.Waveform(Noise)
def SineGaussianFunction(t) :
    return exp(-(t-SineGaussianCenter)**2/(2*sigma**2)) \
        * cos(2*pi*(SineGaussianFreq+SineSweep*(t-SineGaussianCenter)/SineGaussianFWHM)*(t-SineGaussianCenter))
SineGaussian.data = SineGaussianFunction(Noise.Time())
SineGaussianFFT = SineGaussian.dt * np.fft.fft(SineGaussian.data)

## Create a similar Waveform, but offset by some amount to serve as our template
TemplateCenter = 4.0
Template = MatchedFiltering.Waveform(SineGaussian)
Template.data = SineGaussianFunction(Noise.Time()+SineGaussianCenter-TemplateCenter)

## Add the sine-gaussian to the noise to create the signal.
Ratio = 1./10000.
Signal = (1-Ratio)*Noise+Ratio*SineGaussian

## Now, get the quantities computed for the match
OverlapQuantityOfSignal = MatchedFiltering.Overlap(Signal, Template, PSD)
OverlapQuantityOfNoise = MatchedFiltering.Overlap(Noise, Template, PSD)
#OverlapQuantityOfSineGaussian = MatchedFiltering.Overlap(SineGaussian, Template, PSD)
figure()
PlotEvery = 50
semilogy(Noise.Time()[::PlotEvery], OverlapQuantityOfSignal[::PlotEvery], 'g')
semilogy(Noise.Time()[::PlotEvery], OverlapQuantityOfNoise[::PlotEvery], 'b')
#semilogy(Noise.Time()[::PlotEvery], OverlapQuantityOfSineGaussian[::PlotEvery], 'r')

