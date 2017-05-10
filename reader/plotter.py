__author__ = 'cipriancorneanu'

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import itertools
import pandas as pd
import seaborn as sns
import math

def plot_heatmap(axis, mat, labels, show_vals=True):
    # Init panda and seaborn
    sns.set(style="white")
    corr = pd.DataFrame(mat, labels['x'], labels['y'])
    sns.heatmap(corr, annot=True, fmt=".1f", linewidths=.5)

    return axis

def plot_stacked_bar(axis, dt, labels):
    # Plot stacked bar to axis
    ind = np.arange(len(dt))     # the x locations for the groups
    width = 0.35                 # the width of the bars: can also be len(x) sequence

    for i,c in zip(range(0,7), ['b', 'y', 'r', 'g', 'y', 'c']):
        p = axis.bar(ind, [d[i] for d in dt], width, color=c)

    axis.set_ylabel('#Frames')
    axis.set_title('AU intensity distribution')
    axis.set_xticks(ind + width/2., labels)

    return axis

def plot_qualitative(axis, pts=None, ims=None, txts=None):
    # Plot images, scatter of 2D points and text in one axis
    if ims is not None:
        for i, (ax, im) in enumerate(zip(axis, ims)):
            ax.axis('off')
            ax.imshow(im, cmap='Greys', interpolation='nearest')

    if pts is not None:
        for i, (ax, pt) in enumerate(zip(axis, pts)):
            ax.scatter(pt[:,0], pt[:,1])

    if txts is not None:
        for i, (ax, txt) in enumerate(zip(axis, txts)):
            ax.text(10, 10, str(txt), color='r')

    return axis

def plot_t_series(axarr, t_series, labels):
    # Plot list of time series sharing x and y axis
    plt.tick_params(axis='y', which='minor', labelsize=5)

    for ax, ser in zip(axarr, t_series):
        ax.set_ylim([0,5])
        ax.plot(ser['data'])
        ax.set_ylabel(ser['label'], fontsize=8)

    return axarr

def plot_complete_weighted_graph(axis, edge_weights, node_labels):
    n_nodes = len(node_labels)
    G = nx.complete_graph(n_nodes)
    pos = nx.circular_layout(G) # positions for all nodes

    # draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(np.arange(n_nodes)), node_color='r', node_size=1200, alpha=0.8, ax=axis)

    # prepare weighted edges
    edges = [edge for edge in set(itertools.combinations(list(np.arange(n_nodes)),2))]
    weights = [edge_weights[edge] for edge in edges]
    colors = ['r' if w>0 else 'b' for w in weights]

    # norm edges weights' for drawing
    min_r, max_r = min(weights), max(weights)
    min_R, max_R = int(10*min_r), int(10*max_r)
    alpha = (max_R-min_R)/(max_r-min_r)
    weights = [math.floor((x-min_r)*alpha+min_R) for x in weights]

    # draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.7, edge_color=colors, ax=axis)

    # draw labels
    labels = {i: x for i,x in enumerate(node_labels)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=axis)

    # turn axis off
    axis.axis('off')

    return axis

def plot_distribution(axarr, x, dt, labels):
    for ax, d, l in zip(axarr, dt, labels):
        ax.bar(left=x, height=d, width=0.8)
        ax.set_title(l)

    return axarr