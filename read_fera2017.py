__author__ = 'cipriancorneanu'

import getopt
import sys
from reader.dataset import ReaderFera2017
import cPickle
import matplotlib.pyplot as plt

def read_fera2017(argv):
    opts, args = getopt.getopt(argv, '')
    (ipath, opath, mpath, cores) = \
        (
            args[0], args[1], args[2], int(args[3])
        )
    
    fera_ckp = ReaderFera2017(ipath)
    dt = fera_ckp.read(opath, mpath, cores)

if __name__ == '__main__':
    read_fera2017(sys.argv[1:])