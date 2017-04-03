import matplotlib.pyplot as plt
import numpy as np
import cPickle
import itertools
from plotter import *
from processor.partitioner import *

class Explorer():
    def __init__(self, aus=None, au_labels=None, n_int=5, ims=None, lms=None, opath = '../results'):
        self.aus = aus
        self.ims = ims
        self.lms = lms
        self.au_labels = au_labels
        self.n_intensities = n_int
        self.opath = opath

    # TODO: comment better, why 2 cooccurence functions?
    def cooccurrence_intensities(self, au_seq):
        # Pass AUs from ordinal coding to cardinal (5 intensity levels accross 12 AU)
        aus  = [obs[np.nonzero(obs)[0]] + self.n_intensities*np.nonzero(obs)[0] - 1 for obs in au_seq]

        # Compute co-occurences
        coocc = [x for obs in aus for x in itertools.permutations(obs, 2)]

        # Count co-occurences
        dim = au_seq.shape[1] * self.n_intensities
        mat = np.zeros((dim, dim))
        for c in coocc:
            mat[c] += 1

        return mat

    # TODO: why do I have two cooccurence functions?
    def cooccurrence(self, au_seq):
        aus = [np.nonzero(obs)[0] for obs in au_seq if len(np.nonzero(obs)[0])>1]
        coocc = [x for obs in aus for x in itertools.permutations(obs, 2)]

        # Count co-occurences
        dim = au_seq.shape[1]
        mat = np.zeros((dim, dim))
        for c in coocc:
            mat[c] += 1

        # Norm by total number of occurences
        norm = np.sum(mat, axis=1)
        for i in range(0,mat.shape[0]):
            mat[i] = mat[i]/norm[i]

        return mat

    # TODO: What?
    def distribution(self, occ):
        return np.reshape(np.sum(occ, axis=1), (len(self.au_labels),self.n_intensities))

    def distribution_to_stacked_bar(self, distro):
        N = len(np.concatenate(self.aus)) # Number of observations

        distro = [np.hstack((N-np.sum(au), au)) for au in distro]
        distro = [au[::-1] for au in distro]

        return np.reshape([N - np.sum(au[:i]) for au in distro for i in range(0,6)], (12,6))

    def qualitative(self, ims, lms, aus):
        # Prepare qualitative plot of the data
        ims = np.concatenate(self.ims)
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

    def correlation(self, dt):
        return np.corrcoef(dt)

    def label_cardinality(self, labels):
        return np.mean([np.sum(l>0)for l in labels])

    def label_density(self, labels):
        return self.label_cardinality(labels)/(len(self.au_labels)*self.n_intensities)

    def label_diversity(self, labels):
        pass

    def proportion_distinct_label_sets(self, labels):
        pass

if __name__ == '__main__':
    fname = 'disfa'

    path = '/Users/cipriancorneanu/Research/data/disfa/'
    data = cPickle.load(open(path+'disfa_landmarks.pkl', 'rb'))

    explorer = Explorer(
        data['aus'],
        ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU17', 'AU20', 'AU25', 'AU26'],
        5, '../results')

    # Concatenate
    (aus, geom), slices = concat((data['aus'], data['landmarks']))

    # Filter
    aus, slices = filter(aus, slices, lambda x: np.sum(x>0)>1) # Filter AUS

    # Compute exploration
    occ = explorer.cooccurrence(aus)
    occ_int = explorer.cooccurrence_intensities(aus)
    au_distro = explorer.distribution_to_stacked_bar(explorer.distribution(occ_int))
    #ims, lms, aus = explorer.qualitative(data['images'], data['landmarks'], data['aus'])
    corr = explorer.correlation(np.transpose(aus))

    # Plot
    fig1, ax1 = plt.subplots()
    plot_stacked_bar(ax1, au_distro, explorer.au_labels)
    plt.savefig('../results/au_distribution.png')

    fig2, ax2 = plt.subplots()
    plot_heatmap(ax2, corr, labels={'x':explorer.au_labels, 'y':explorer.au_labels})
    plt.savefig('../results/au_correlation.png')

    fig3, ax3 = plt.subplots()
    plot_complete_weighted_graph(ax3, explorer.au_labels, corr)
    plt.savefig('../results/au_correlations_graph.png')

    # Plot distributions
    fig4, axarr4 = plt.subplots(1,12)
    plot_distribution(axarr4, x=[0,1,2,3,4], dt=explorer.distribution(occ_int), labels=explorer.au_labels)
    plt.savefig('../results/au_intensity_distribution.png')

    print 'label cardinality : {}'.format(explorer.label_cardinality(aus))

    print 'label density : {}'.format(explorer.label_density(aus))

    # Plot time series
    print ("Plot AU dynamics")
    for i,x in enumerate(data['aus']):
        t_series = [{'data':col,'label':lab} for col,lab in zip(x.T, explorer.au_labels) if np.sum(col)>0]

        fig3, axarr3 = plt.subplots(len(t_series), 1, figsize=(8,8))
        plot_t_series(axarr3, t_series, explorer.au_labels)

        # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
        fig3.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig3.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in fig3.axes], visible=False)

        fig3.savefig('../results/au_temp_dynamics_' + str(i) + '.png')