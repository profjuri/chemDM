# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:11:08 2021

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

    x1 = df['LUMO']
    x2 = df['HOMO']

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])

    n, bins, patches = ax.hist(x2, bins=50, histtype='bar', rwidth=0.7,
                               color="darkgreen", alpha=0.5, label='HOMO')

    n, bins, patches = ax.hist(x1, bins=50, histtype='bar', rwidth=0.7, color="darkblue", alpha=0.5, label='LUMO')

    ax.set_xlabel(r'$E$ / eV', size=32)
    ax.set_ylabel(r'Count', size=32)

    xtickmaj = ticker.MultipleLocator(2)
    xtickmin = ticker.AutoMinorLocator(4)
    ytickmaj = ticker.MultipleLocator(1000)
    ytickmin = ticker.AutoMinorLocator(4)
    ax.xaxis.set_major_locator(xtickmaj)
    ax.xaxis.set_minor_locator(xtickmin)
    ax.yaxis.set_major_locator(ytickmaj)
    ax.yaxis.set_minor_locator(ytickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='major', direction='in', labelsize=28, pad=10, length=5)
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=28, pad=10, length=2)

    ax.grid(color='black', which='major', axis='y', linestyle='dashed', alpha=0.3)

    plt.legend(loc='upper right', frameon=False, fontsize=20)

    # plt.savefig("csd_fig4.pdf", dpi=300)
    plt.show()
