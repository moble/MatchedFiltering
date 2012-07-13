def ReadWAVFile(wave_file, channel=0) :
    """
    Read a given WAV file.
    
    The returned data is a numpy array of floats in the range [-1,1),
    along with the sample rate (samples per second), and the total
    number of samples.
    
    You might call this function with something like
    >>> WAVData, SampleRate, NumSamples = ReadWAVFile('test.wav')
    
    The additional optional argument 'channel' might be used to pick
    out one channel if the WAV data is stereo.  The default is to just
    return the first channel.
    """
    
    import wave
    import struct
    from numpy import array
    
    stream = wave.open(wave_file,"rb")
    
    num_channels = stream.getnchannels()
    sample_rate = stream.getframerate()
    sample_width = stream.getsampwidth()
    num_frames = stream.getnframes()
    
    raw_data = stream.readframes( num_frames ) # Returns byte data
    stream.close()
    
    total_samples = num_frames * num_channels
    
    if sample_width == 1 :
        fmt = "%iB" % total_samples # read unsigned chars
        offset = 128
        scale = 128.0
    elif sample_width == 2 :
        fmt = "%ih" % total_samples # read signed 2 byte shorts
        offset = 0
        scale = 32768.0
    elif sample_width == 4 :
        fmt = "%ii" % total_samples # read signed 4 byte ints
        offset = 0
        scale = 2147483648.0
    else :
        raise ValueError("Only supports 8, 16, and 32 bit audio formats.")
    
    raw_data = (array(struct.unpack(fmt, raw_data), dtype=float)-offset) / scale
    
    raw_data = raw_data[channel::num_channels]
    
    return raw_data, sample_rate, num_frames
