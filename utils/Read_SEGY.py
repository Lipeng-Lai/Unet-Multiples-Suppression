import segyio
import numpy as np

def ReadSEGYData(filename):
    with segyio.open(filename, "r", ignore_geometry=True)as f:
        f.mmap()
        nTrace = f.tracecount
        nSample = f.bin[segyio.BinField.Samples]
        startT = 0
        deltaT = f.bin[segyio.BinField.Interval]
        print("     Number of Trace   = %d" % (nTrace))
        print("     Number of Samples = %d" % (nSample))
        print("     Start Samples     = %d" % (startT))
        print("     Sampling Rate     = %d" % (deltaT))
        data2D = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data2D