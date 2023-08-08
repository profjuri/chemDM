# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:11:08 2021

@author: Omer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec
from matplotlib import rc
import matplotlib

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def dist(df, colors=None, integral=True):

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(1, 1)

    if not colors:
        colors = ["gold"]

    names = df.columns
    ax = plt.subplot(gs[0])
    if integral:
        ax1 = ax.twinx()

    nbins = np.arange(np.ceil(df.min().min()), np.floor(df.max().max()), 0.2)
    for i in range(df.shape[1]):

        clr = colors[i]
        data = df.iloc[:,i].dropna()
        # nbins = int(np.log2(len(data)) + 1) * 4
        # nbins = int(np.log2(len(data)) + 1) * 4
        title = ' '.join(names[i].split("_"))
        hist, bins, _ = ax.hist(data, bins=nbins, histtype='bar', rwidth=0.75,
                                hatch='//', fill=False, color=clr,
                                edgecolor=clr, label=title)

        integrals = np.cumsum(hist)
        if integral:
            ax1.plot(bins[1:], integrals, color=clr)

    # ax.axvspan(4, 5.5, alpha=0.5, color="gray")
    ax.legend(loc=2, fontsize=16).draw_frame(False)
    ax.set_xlabel("Energy / eV", size=20)
    ax.set_ylabel("Count", size=20)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=16, pad=10, length=7)
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=16, pad=10, length=3)
    tickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(tickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('left')
    ax.minorticks_on()
    ax.set_xlim(0, 10)

    if integral:
        ax1.set_ylim(0)
        ax1.set_ylabel("Integral", size=20)
        ax1.tick_params(axis='y', which='major', direction='in', labelsize=16, pad=10, length=7)
        ax1.tick_params(axis='both', which='minor', direction='in', labelsize=16, pad=10, length=3)
        ax1.yaxis.set_ticks_position('right')
        ax1.minorticks_on()

    return



if __name__ == '__main__':

    
    ## PM7 plot
    df = pd.read_csv(r"PM7_funnel.csv", index_col=0)
    df['GAP cal PM7'] = df['LUMO_pm7_cal'] - df['HOMO_pm7_cal']
    x1 = df[['GAP cal PM7']]    
    dist(x1, colors=["gold"])
    # plt.savefig("csd_fig_dist_pm7.svg", dpi=600, bbox_inches='tight')
    
    ## 3-21 plot
    df1 = pd.read_csv(r"basis_funnel.csv", index_col=0)
    df1['GAP cal B3'] = df1['LUMO_321_cal'] - df1['HOMO_321_cal']
    x2 = df1[['GAP cal B3']]    
    dist(x2, colors=["orange"])
    # plt.savefig("csd_fig_dist_B3.svg", dpi=600, bbox_inches='tight')
    plt.show()
