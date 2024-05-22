#!/usr/bin/env python
import numpy as np
import ulysses
import multiprocessing
import random

__doc__="""

Scan of EtaB in two variables

%prog -m 1DME --ordering 0 --loop -o scan.pdf  PARAMETERFILE -x 100 -y 100

Example paramter file:

m     -3   -1 # logarithm, in [ev]
M1    11   14 # logarithm, in [GeV]
M2    12.6    # logarithm, in [GeV]
M3    13      # logarithm, in [GeV]
delta 213     # [deg]
a21    81     # [deg]
a31   476     # [deg]
x1     90     # [deg]
x2     87     # [deg]
x3    180     # [deg]
y1   -120     # [deg]
y2      0     # [deg]
y3   -120     # [deg]
t12    33.63  # [deg]
t13     8.52  # [deg]
t23    49.58  # [deg]
"""


if __name__=="__main__":

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-o", "--output",    dest="OUTPUT",      default="scan.pdf", type=str, help="Output file name for evolution plots/data (default: %default)")
    op.add_option("-v", "--debug",     dest="DEBUG",       default=False, action="store_true", help="Turn on some debug messages")
    op.add_option("-m", "--model",     dest="MODEL",       default="1DME", help="Selection of of model (default: %default)")
    op.add_option("-n", "--n-scan",   dest="NSCAN",      default=30, type=int, help="Number of point to scan (default: %default)")
    op.add_option("--zrange",          dest="ZRANGE",      default="0.1,100,1000", help="Ranges and steps of the evolution variable (default: %default)")
    op.add_option("--inv",             dest="INVORDERING", default=False, action='store_true', help="Use inverted mass ordering (default: %default)")
    op.add_option("--loop",            dest="LOOP",        default=False, action='store_true', help="Use loop-corrected Yukawa (default: %default)")
    opts, args = op.parse_args()


    if len(args)==0:
        print("No parameter space configuration given, exiting.")
        sys.exit(1)

    # Make sure specified file actually exists
    if not os.path.exists(args[0]):
        print("Specified input file {} does not exist, exiting.".format(args[0]))

    # Disect the zrange string
    zmin, zmax, zsteps = opts.ZRANGE.split(",")
    zmin=float(zmin)
    zmax=float(zmax)
    zsteps=int(zsteps)

    assert(zmin<zmax)
    assert(zsteps>0)

    pfile, gdict = ulysses.tools.parseArgs(args)

    LEPTO = ulysses.selectModel(opts.MODEL,
            zmin=zmin, zmax=zmax, zsteps=zsteps,
            ordering=int(opts.INVORDERING),
            loop=opts.LOOP,
            debug=opts.DEBUG,
            **gdict
            )

    # Read parameter card and very explicit checks on parameter names
    RNG, FIX, isCasIb = ulysses.readConfig(pfile)

    
    print("Sparse Scanning for {} values.".format(opts.NSCAN))

    #from progressbar import ProgressBar
    #pbar = ProgressBar()
    
    from tqdm import tqdm
    
    #for px in tqdm(PPX):
    #    for py in tqdm(PPY, leave = False):
    #        FIX[pxscan] = px
    #        FIX[pyscan] = py
    #        etaB = LEPTO(FIX)
    #        EE.append(etaB)

    DATA = []
    
    #for i in range(opts.NSCANX):
    #    for j in range(opts.NSCANY):
    #        DATA[i*opts.NSCANY + j, 0] = PPX[i]
    #        DATA[i*opts.NSCANY + j, 1] = PPY[j]
    #        DATA[i*opts.NSCANY + j, 2] = EE[i*opts.NSCANY + j]

    #for i in range(opts.NSCANX):
    #    for j in range(opts.NSCANY):
    #        xList.append([i,j,PPX[i],PPY[j]])

    def genData(i):
        tempFIX = FIX

        x1 = random.randint(0,180)
        y1 = random.randint(0,180)
        x2 = random.randint(0,180)
        y2 = random.randint(0,180)
        x3 = random.randint(0,180)
        y3 = random.randint(0,180)

        tempFIX["x1"] = x1
        tempFIX["y1"] = y1
        tempFIX["x2"] = x2
        tempFIX["y2"] = y2
        tempFIX["x3"] = x3
        tempFIX["y3"] = y3

        try:
            etaB = LEPTO(FIX)
        except:
            print("An exception occurred.")
        return [x1,y1,x2,y2,x3,y3,etaB]


    pool = multiprocessing.Pool()
    values = [(i) for i in range(opts.NSCAN)]
    #results=pool.map(genData_wrapper, values)
    results = list(tqdm(pool.imap(genData, values), total=opts.NSCAN))
    pool.close()

    for i in range(opts.NSCAN):
        if results[i][6] > 5.8*10**(-10):
            DATA.append([(results[i][j]) for j in range(7)])

    print(DATA)

    if opts.OUTPUT is not None:
        if opts.OUTPUT.endswith(".txt"):
            np.savetxt(opts.OUTPUT, DATA)
        elif opts.OUTPUT.endswith(".csv"):
            np.savetxt(opts.OUTPUT, DATA, delimiter=",")
        if opts.DEBUG:
            print("Output written to {}".format(opts.OUTPUT))