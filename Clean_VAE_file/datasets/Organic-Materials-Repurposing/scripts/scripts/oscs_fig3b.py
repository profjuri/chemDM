# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:57:35 2021

@author: Omer
"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec
from matplotlib import rc
import matplotlib

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


if __name__ == '__main__':

    df = pd.read_csv(r"CSD_EES_DB.csv", index_col=0)

    x0 = df['E(S1)']
    x1 = df['E(S2)']
    x2 = df['E(T1)']
    x3 = df['E(T2)']

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])

    n, bins, patches = ax.hist(x0, bins=95, histtype='stepfilled', rwidth=1,
                               fill=False, alpha=0.6, edgecolor='cyan', label='S$_1$')
    n, bins, patches = ax.hist(x1, bins=85, histtype='stepfilled', rwidth=1,
                                fill=False,alpha=0.6, edgecolor='lightseagreen', label='S$_2$')
    n, bins, patches = ax.hist(x2, bins=85, histtype='stepfilled', rwidth=1,
                               fill=False, alpha=0.6, edgecolor='red', label='T$_1$')
    n, bins, patches = ax.hist(x3, bins=85, histtype='stepfilled', rwidth=1,
                               fill=False, alpha=0.6, edgecolor='tomato', label='T$_2$')


    ax.set_xlabel(r'$E$ / eV', size=32)
    ax.set_ylabel(r'Count', size=32)

    xtickmaj = ticker.MultipleLocator(1)
    xtickmin = ticker.AutoMinorLocator(4)
    ytickmaj = ticker.MultipleLocator(500)
    ytickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_major_locator(xtickmaj)
    ax.xaxis.set_minor_locator(xtickmin)
    ax.yaxis.set_major_locator(ytickmaj)
    ax.yaxis.set_minor_locator(ytickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='major', direction='in', labelsize=28, pad=10, length=5)
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=28, pad=10, length=2)

    ax.set_xlim(0, 6)
    ax.grid(color='black', which='major', axis='y', linestyle='--', alpha=0.2)

    plt.legend(loc='upper right', frameon=False, fontsize=20)

    # plt.savefig("csd_fig1.pdf", dpi=300)
    plt.show()
