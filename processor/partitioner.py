__author__ = 'cipriancorneanu'
import cPickle
import numpy as np
import matplotlib.pyplot as plt

def fold(N, n):
    train  = [[y for y in range(0,N) if y not in range(x*int(N/n), (x+1)*int(N/n))] for x in range(0,n)]
    test = [range(x*int(N/n), (x+1)*int(N/n)) for x in range(0,n)]

    return (train, test)

def slice(list):
    lengths = [len(x) for x in list]
    acc_lengths = [int(np.sum(lengths[:i])) for i in range(0,len(lengths)+1)]

    return np.asarray([np.arange(start, stop) for start,stop in zip(acc_lengths[:-1], acc_lengths[1:])])

def concat(dt):
    return(
        tuple([np.concatenate(x) for x in dt]),
        slice(dt[0])
    )

def deconcat(array, slices):
    return [np.asarray(array[slice]) for slice in slices]

def filter(aus, slices, condition):
    aus = deconcat(aus, slices)

    slices_filtered = np.asarray(
            [np.transpose([np.where(np.asarray([condition(x) for x in a], dtype=np.uint8)==True)[0]]) for a in aus]
        )

    return (
        np.squeeze(np.concatenate([a[s] for a, s in zip(aus, slices_filtered)])),
        np.asarray([sl[0]+slf  for sl, slf in zip(slices, slices_filtered)])
    )

def transform(array):
    return [1 for a in array if a>0]

def vectorize(aus):
    n_levels, n_labels = 6, aus.shape[1]
    out = np.zeros((aus.shape[0], n_labels*n_levels), dtype=np.uint8)

    for i,a in enumerate(aus):
        activate = np.asarray([x+y for x,y in zip(a, np.arange(0,n_labels)*n_levels) if x>0])
        if len(activate)>0:
            out[i,activate[activate>0]] = 1
    return out

def vectorize_(aus):
    return np.asarray([x>1 for x in aus]).astype(int)

def merge(aus, slices):
    # Merge highest intensity labels
    pass

if __name__ == '__main__':
    # Load data
    path = '/Users/cipriancorneanu/Research/data/disfa/'
    dt = cPickle.load(open(path+'disfa.pkl', 'rb'))

    # Concatenate
    (aus, geom), slices = concat((dt['aus'], dt['landmarks']))

    # Filter 1
    aus1, slices1 = filter(aus,slices, lambda x: x[2]>3)
    geom1 = np.squeeze(geom[np.concatenate(slices1)])

    # Filter 2
    aus2, slices2 = filter(aus,slices, lambda x: x[6]>3)
    geom2 = np.squeeze(geom[np.concatenate(slices2)])
    print geom1.shape
    print geom2.shape

    # Encode
    #encoded_geom = encode_parametric(geom)

    # Plot
    for g1, g2 in zip(geom1[:10],geom2):
        plt.plot(g1[:,0], g1[:,1])
        plt.plot(g2[:,0], g2[:,1])

    plt.savefig('/Users/cipriancorneanu/Research/code/afea/results/geoms.png')