__author__ = 'cipriancorneanu'

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def f1(gt, est):
    f1_score(gt, est, average='micro')

def accuracy(gt, est):
    accuracy_score(gt, est)

def mse(gt, est):
    return np.mean((gt-est)**2, axis=0)

def rmse(gt, est, all=True):
    return np.mean((gt-est)**2, axis=0)**0.5

def icc(gt, est, cas=3, typ=1):
    #y_hat, y_lab = _pre_process(y_hat, y_lab)

    gt = np.transpose(gt)
    est = np.transpose(est)

    Y = np.array((gt, est))

    # number of targets
    n = Y.shape[2]

    # mean per target
    mpt = np.mean(Y, 0)

    # print mpt.eval()
    mpr = np.mean(Y, 2)

    # print mpr.eval()
    tm = np.mean(mpt, 1)

    # within target sum sqrs
    WSS = np.sum((Y[0]-mpt)**2 + (Y[1]-mpt)**2, 1)

    # within mean sqrs
    WMS = WSS/n

    # between rater sum sqrs
    RSS = np.sum((mpr - tm)**2, 0) * n

    # between rater mean sqrs
    RMS = RSS

    # between target sum sqrs
    TM = np.tile(tm, (est.shape[1], 1)).T
    BSS = np.sum((mpt - TM)**2, 1) * 2

    # between targets mean squares
    BMS = BSS / (n - 1)

    # residual sum of squares
    ESS = WSS - RSS

    # residual mean sqrs
    EMS = ESS / (n - 1)

    if cas == 1:
        if typ == 1:
            res = (BMS - WMS) / (BMS + WMS)
        if typ == 2:
            res = (BMS - WMS) / BMS
    if cas == 2:
        if typ == 1:
            res = (BMS - EMS) / (BMS + EMS + 2 * (RMS - EMS) / n)
        if typ == 2:
            res = (BMS - EMS) / (BMS + (RMS - EMS) / n)
    if cas == 3:
        if typ == 1:
            res = (BMS - EMS) / (BMS + EMS)
        if typ == 2:
            res = (BMS - EMS) / BMS

    res[np.isnan(res)] = 0
    return res.astype('float32')

def corr(gt, est):
    return np.corrcoef(gt,est)



