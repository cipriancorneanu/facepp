__author__ = 'cipriancorneanu'

import getopt
import sys
from reader.dataset import ReaderFera2017

def read_fera2017(ipath, opath, mpath):
    '''
    opts, args = getopt.getopt(argv, '')
    (ipath, mpath) = \
        (
            args[0], args[1]
        )
    '''

    fera_ckp = ReaderFera2017(ipath)
    dt = fera_ckp.read(ipath, opath, mpath)

if __name__ == '__main__':
    #read_fera2017(sys.argv[1:])

    read_fera2017(
        '/Users/cipriancorneanu/Research/data/fera2017/train/',
        '/Users/cipriancorneanu/Research/data/fera2017/results/',
        '/Users/cipriancorneanu/Research/code/facepp/models/'
    )