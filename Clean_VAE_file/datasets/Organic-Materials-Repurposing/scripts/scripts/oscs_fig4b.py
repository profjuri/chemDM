# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:17:08 2021

@author: Omer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec
from matplotlib import rc
import matplotlib

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

mpl.rcParams['agg.path.chunksize'] = 10000


if __name__ == '__main__':

    df = pd.read_csv(r"OSCs_db.csv", index_col=0)

    x = df['E(S1)']
    y = df['E(S2)']

    df['log(f(S2))'] = np.log(df['f(S2)'])

    z = df['log(f(S2))']

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.15, hspace=0.30)
    
    ax = plt.subplot(gs[0])
    
    plt.scatter(x, y, c=z, cmap=plt.cm.hot, alpha=0.6, marker='.') 

    ax.set_xlabel('S$_1$ / eV', size=32)
    ax.set_ylabel('S$_2$ / eV', size=32)
    
    xtickmaj = ticker.MultipleLocator(1)
    xtickmin = ticker.AutoMinorLocator(4)
    ytickmaj = ticker.MultipleLocator(1)
    ytickmin = ticker.AutoMinorLocator(4)
    ax.xaxis.set_major_locator(xtickmaj)
    ax.xaxis.set_minor_locator(xtickmin)
    ax.yaxis.set_major_locator(ytickmaj)
    ax.yaxis.set_minor_locator(ytickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='major', direction='in', labelsize=28, pad=10, length=5)
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=28, pad=10, length=2)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    cbar = plt.colorbar()
    cbar.ax.tick_params(size=0, labelsize=22)

    cbar_max = 0.33
    cbar_min = 0
    cbar_step = 0.03
    cbar_num_colors = 200
    cbar_num_format = "%d"
    
    #cbar.ax.get_yaxis().set_visible(False)
    cbar.ax.get_yaxis().labelpad = 25
    cbar.set_label('log($f_{S_2}$)', rotation=90, size=32)

    x = np.arange(-1, 7, 0.01)
    plt.plot(x, 2 * x, linestyle='--', c='black', linewidth=2)

    plt.savefig("csd_fig3.pdf", dpi=300)
    # plt.show()
