import neo 

def read_ns5_file(filename):
    """ Function to read files of .ns5 format and return the data and time values."""
    reader = neo.io.BlackrockIO(filename = filename, verbose = True)
    times = reader.read_block(0).segments[0].analogsignals[0].times 
    data = reader.read_block(0).segments[0].analogsignals[0].magnitude

    return times, data