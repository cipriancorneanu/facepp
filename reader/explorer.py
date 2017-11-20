import matplotlib.pyplot as plt
import numpy as np
import cPickle
import itertools
from plotter import *
from facepp.processor.partitioner import *

# Intensity cooccurence
def cooccurrence_intensities(dt):
    # Pass AUs from ordinal coding to cardinal (n_intensities intensity levels across n_labels)
    n_intensities = len(set(list(np.reshape(np.concatenate(dt), -1))))

    aus  = [obs[np.nonzero(obs)[0]] + n_intensities*np.nonzero(obs)[0] for obs in dt]

    # Compute co-occurences
    coocc = [x for obs in aus for x in itertools.permutations(obs, 2)]

    # Count co-occurences
    dim = dt.shape[1] * n_intensities
    mat = np.zeros((dim, dim))
    for c in coocc:
        mat[c] += 1

    return mat

# Binary cooccurence
def cooccurrence(dt):
    dt_bin = [np.nonzero(obs)[0] for obs in dt if len(np.nonzero(obs)[0])>1]
    coocc = [x for obs in dt_bin for x in itertools.permutations(obs, 2)]

    # Count co-occurences
    dim = dt.shape[1]
    mat = np.zeros((dim, dim))
    for c in coocc:
        mat[c] += 1

    # Norm by total number of occurences
    norm = np.sum(mat, axis=1)
    for i in range(0,mat.shape[0]):
        mat[i] = mat[i]/norm[i]

    return mat

def distribution(dt, labels):
    n_intensities = len(set(list(np.reshape(np.concatenate(dt), -1))))
    return np.reshape(np.sum(dt, axis=1), (len(labels), n_intensities))

def distribution_to_stacked_bar(aus, distro):
    N = len(np.concatenate(aus)) # Number of observations

    distro = [np.hstack((N-np.sum(au), au)) for au in distro]
    distro = [au[::-1] for au in distro]

    return np.reshape([N - np.sum(au[:i]) for au in distro for i in range(0,6)], (12,6))

def qualitative(ims, lms, aus):
    # Prepare qualitative plot of the data
    ims = np.concatenate(ims)
    lms = np.concatenate(lms)
    aus = np.concatenate(aus)

    # Pull N frames with with at least one AU of predefined intensity
    filtered = np.asarray([i for i, a in enumerate(aus) if 3 in a])

    # If too many pull N randomly
    if len(filtered) > 10: filtered = filtered[(len(filtered)*np.random.rand(10)).astype(int)]

    # Pull output
    ims = ims[filtered]
    lms = lms[filtered]
    #aus = [''.join([self.au_labels[item] for item in np.nonzero(a)[0]]) for a in aus[filtered]]
    aus = aus[filtered]

    return ims, lms, aus

def correlation(dt):
    return np.corrcoef(dt)

def label_cardinality(labels):
    return np.mean([np.sum(l>0) for l in labels])

def label_density(labels, au_labels):
    n_intensities = len(set(list(np.reshape(np.concatenate(labels), -1))))
    return label_cardinality(labels)/(len(au_labels)*n_intensities)

def label_diversity(labels):
    pass

def proportion_distinct_label_sets(labels):
    pass



if __name__ == '__main__':
    fname = 'disfa'

    path = '/Users/cipriancorneanu/Research/data/'
    opath = '/Users/cipriancorneanu/Research/code/afea/'

    data = cPickle.load(open(path+'bp4d_labels.pkl', 'rb'))

    aus = np.reshape(data, (-1, 12))

    au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']

    # Concatenate
    #aus, slices = concat(data['aus'])

    # Filter
    '''
    aus, slices = filter(aus, slices, lambda x: np.sum(x>0)>1) # Filter AUS
    print aus.shape
    '''

    # Compute exploration
    #occ = explorer.cooccurrence(aus)
    occ_int = cooccurrence_intensities(aus)
    #au_distro = distribution_to_stacked_bar(distribution(occ_int))
    #ims, lms, aus = explorer.qualitative(data['images'], data['landmarks'], data['aus'])
    corr = correlation(np.transpose(aus))

    # Plot
    '''
    fig1, ax1 = plt.subplots()
    plot_stacked_bar(ax1, au_distro, au_labels)
    plt.savefig('../results_disfa/au_distribution.png')
    '''

    fig2, ax2 = plt.subplots()
    plot_heatmap(ax2, corr, labels={'x':au_labels, 'y':au_labels})
    plt.savefig(opath+'/results/au_correlation.eps')

    fig3, ax3 = plt.subplots()
    plot_complete_weighted_graph(ax3, corr, au_labels,)
    plt.savefig(opath+'/results/au_correlations_graph.eps')

    # Plot distributions
    '''
    fig4, axarr4 = plt.subplots(1,12)
    plot_distribution(axarr4, x=[0,1,2,3,4], dt=distribution(occ_int), labels=au_labels)
    plt.savefig('../results_disfa/au_intensity_distribution.png')
    '''

    print 'label cardinality : {}'.format(label_cardinality(aus))

    print 'label density : {}'.format(label_density(aus, au_labels))

    # Plot time series
    '''
    print ("Plot AU dynamics")
    for i,x in enumerate(data['aus']):
        t_series = [{'data':col,'label':lab} for col,lab in zip(x.T, au_labels) if np.sum(col)>0]

        fig3, axarr3 = plt.subplots(len(t_series), 1, figsize=(8,8))
        plot_t_series(axarr3, t_series, au_labels)

        # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
        fig3.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig3.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in fig3.axes], visible=False)

        fig3.savefig('../results_disfa/au_temp_dynamics_' + str(i) + '.png')
    '''