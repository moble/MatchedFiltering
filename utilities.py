import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

max_safe_exponent = np.log(2)*(np.finfo(float).maxexp-1)


def transition_function(t, t0, t1):
    """Smooth function going from 0 before t0 to 1 after t1, with C^infty transition in between"""
    assert t0 <= t1
    f = np.zeros_like(t)
    if t0 >= t[-1]:
        return f
    f[t >= (t0+t1)/2] = 1.0
    if t1 <= t[0]:
        return f
    transition_indices = (t0 < t) & (t < t1)
    transition = (t[transition_indices]-t0) / (t1 - t0)
    transition = 1.0/transition - 1.0/(1.0-transition)
    safe_indices = (transition < max_safe_exponent)
    transition_indices[transition_indices] = safe_indices
    f[transition_indices] = 1.0 / (1.0 + np.exp(transition[safe_indices]))
    return f


def bump_function(t, t0, t1, t2, t3):
    """Smooth (C^infty) function going from 0 before t0 to 1 between t1 and t2, then back to 0 after t3
    
    This function is based on `transition_function`.
    """
    return transition_function(t, t0, t1) * (1 - transition_function(t, t2, t3))


def fade(signal, fade_length=0.075):
    """Fade a signal in at the begining, and out at the end
    
    This uses `bump_function` to smoothly fade from zero to full signal over the fraction `fade_length`
    of the data.  The output data is then precisely equal to the input until `fade_length` before the
    end, where it is smoothly faded back to zero.  This is useful for making sure there are no harsh
    noises when a signal begins and ends, and also reduces noise due to convolution with sharp filters.
    
    """
    n = len(signal)
    t = np.arange(n, dtype=float)
    return signal * bump_function(t, t[0], t[int(fade_length*n)], t[int(-1-fade_length*n)], t[-1])


def filter_and_plot(h, t, htilde, sampling_rate, sliders, notch_filters, use_equalizer, frequencies, frequency_bin_upper_ends, hide_original_time_domain):
    from IPython.display import display, clear_output, Audio

    # Once new output is available, clear the old output
    clear_output(wait=True)

    # Get levels from sliders
    levels = np.ones_like(frequencies)
    if use_equalizer.value is True:
        slider_values = [s.value for s in sliders]
        for i, f in enumerate(frequency_bin_upper_ends):
            if i==0:
                f_last = 0.0
            levels[(frequencies >= f_last) & (frequencies < f)] = 10**slider_values[i]
            f_last = f

    # Get notch filters (if any)
    for notch_filter in notch_filters.children:
        f_begin, f_end, f_bool = [child.value for child in notch_filter.children if not isinstance(child, widgets.Label)]
        if (f_bool is True) and (f_begin<f_end):
            levels[(frequencies >= f_begin) & (frequencies < f_end)] = 0.0
    
    # Filter the data and transform back to the time domain
    hprime = sampling_rate * np.fft.irfft(htilde*levels)
    
    # Smooth the beginning and end, so there are no loud spikes as the audio turns on and off
    hprime = fade(hprime, 0.05)
    
    if hide_original_time_domain.value:
        plot_td_and_fd(t, hprime, frequencies, htilde*levels, h=None, htilde=htilde)
    else:
        plot_td_and_fd(t, hprime, frequencies, htilde*levels, h=h, htilde=htilde)


def plot_td_and_fd(t, hprime, f, htildeprime, h=None, htilde=None):
    from IPython.display import display, clear_output, Audio
    print('Contrast: {0:.4f}'.format(np.max(np.abs(hprime)) / np.sqrt(np.mean(np.abs(hprime)**2))))
    sampling_rate = 1.0/(t[1]-t[0])
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if h is not None:
        ax1.plot(t, h, label='Raw data')
        ax1.plot(t, hprime, label='Filtered data')
        ax1.legend(loc='lower left');
    else:
        ax1.plot(t, hprime)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Detector strain $h$ (dimensionless)')
    ax1.set_xlim(xmax=t[-1])
    ax1.set_ylim(1.1*np.min(hprime), 1.1*np.max(hprime))
    ax1.set_title('Time domain')
    ax1.grid()
    if htilde is not None:
        ax2.loglog(f, abs(htilde), label='Raw data')
        ax2.loglog(f, abs(htildeprime), label='Filtered data')
        ax2.legend(loc='lower left');
    else:
        ax2.loglog(f, abs(htildeprime))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel(r'Detector strain Fourier transform $\tilde{h}$ (seconds)')
    ax2.set_xlim(1, sampling_rate/2)
    ax2.set_title('Frequency domain')
    ax2.grid()
    fig.tight_layout()
    display(Audio(data=hprime, rate=int(sampling_rate), autoplay=False))
    return fig, (ax1, ax2)


def add_notch_filter(notch_filters, gap_filler):
    """Add a button for a simple square notch filter"""
    new_filter = widgets.HBox([widgets.FloatText(description='Begin', width='150px'),
                               widgets.FloatText(description='End', width='150px'),
                               gap_filler,
                               widgets.Checkbox(description='Use filter', value=True)])
    notch_filters.children += (new_filter,)

    
def filter_cheat(frequency_bin_upper_ends, sliders, notch_filters, gap_filler, cheat_sliders=True, cheat_notches=True):
    if cheat_sliders:
        for f,s in zip(frequency_bin_upper_ends, sliders):
            if f<63 or f>257:
                s.value = -15.0
    if cheat_notches:
        notch_filters.children = tuple(widgets.HBox([widgets.FloatText(value=e, description='Begin', width='150px'),
                                                     widgets.FloatText(value=b, description='End', width='150px'),
                                                     gap_filler,
                                                     widgets.Checkbox(description='Use filter', value=True)])
                                       for e,b in [(59.8, 60.2), (119.5, 120.5), (179.0, 181.0), (299., 304.), (331.4, 334.0)])


def notch_data(h, sampling_rate, notch_locations_and_sizes):
    from scipy.signal import iirdesign, zpk2tf, filtfilt
    nyquist_frequency = 0.5 * sampling_rate
    h_filtered = h.copy()
    for i, (notch_low, notch_high, size) in enumerate(notch_locations_and_sizes):
        notch_width = notch_high - notch_low
        pass_low = (notch_low - 2*notch_width) / nyquist_frequency
        pass_high = (notch_high + 2*notch_width) / nyquist_frequency
        notch_low = notch_low / nyquist_frequency
        notch_high = notch_high / nyquist_frequency
        b, a = zpk2tf(*iirdesign([pass_low, pass_high], [notch_low, notch_high], gpass=size/4.0, gstop=size, output='zpk'))
        h_filtered = filtfilt(b, a, h_filtered)
    return h_filtered


def whiten(signal, sampling_rate):
    from numpy.fft import rfft, irfft, rfftfreq
    from scipy.signal import welch
    from scipy.interpolate import InterpolatedUnivariateSpline
    f_psd, psd = scipy.signal.welch(signal, sampling_rate, nperseg=2**int(np.log2(len(signal)/8.0)), scaling='density')
    f_signal = rfftfreq(len(signal), 1./sampling_rate)
    psd = np.abs(InterpolatedUnivariateSpline(f_psd, psd)(f_signal))
    return irfft(rfft(signal) / np.sqrt(0.5 * sampling_rate * psd))


def bandpass(signal, sampling_rate, lower_end=20.0, upper_end=300.0):
    from scipy.signal import butter, filtfilt
    nyquist_frequency = sampling_rate/2.0
    bb, ab = butter(4, [lower_end/nyquist_frequency, upper_end/nyquist_frequency], btype='band')
    return filtfilt(bb, ab, signal)
