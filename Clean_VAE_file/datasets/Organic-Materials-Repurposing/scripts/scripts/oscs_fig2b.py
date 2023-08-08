# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:54:38 2021

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

    df = pd.read_csv(r"OSCs_date_hist.csv")
    df = df.reset_index(0)
    
    x = df['year']
    y = df['frac_per']
    
    
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])

    ax.bar(x, y, alpha=0.6, color='darkgreen') 

    ax.set_xlabel('Year', size=22)
    ax.set_ylabel('Fraction of OSCs deposited / \%', size=22)

    xtickmaj = ticker.MultipleLocator(10)
    xtickmin = ticker.AutoMinorLocator(2)
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

    plt.show()
    # plt.savefig("csd_osc_frac.svg", dpi=600)
