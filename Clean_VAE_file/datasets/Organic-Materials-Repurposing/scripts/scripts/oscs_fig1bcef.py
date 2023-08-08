#!/usr/bin/env python

import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import normaltest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib import rc
import matplotlib

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def residuals(df, xname, yname):

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(1, 1)

    # Linear Regression
    df_clean = df.dropna(subset=[xname, yname])
    x = df_clean[xname].values.reshape(-1,1)
    y = df_clean[yname]
    regr = LinearRegression()
    regr.fit(x, y)
    y_hat = regr.predict(x)
    rmse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    m = regr.coef_
    b = regr.intercept_
    residuals = y - y_hat
    p = normaltest(residuals)[1]

    if residuals.max() > 1.0:
        bins_p = np.arange(0, residuals.max(), 0.25)
    else:
        nbins = int((np.log2(len(residuals)) + 1) / 2)
        bins_p = np.linspace(0, residuals.max(), nbins)

    bins_n = (-1 * bins_p[1:])[::-1]
    bins = np.r_[ bins_n, bins_p ]

    xtit = xname.split("_")
    xtit = ' '.join([ xtit[0], xtit[-1] ])
    ytit = yname.split("_")
    ytit = ' '.join([ ytit[0], ytit[-1] ])

    ax = plt.subplot(gs[0])
    ax.hist(residuals, bins=bins, histtype='bar', rwidth=0.75, hatch='//',
            fill=False, color="b", edgecolor="b", label=u'%s vs. %s' % (xtit, ytit))

    ax.legend(fontsize=16).draw_frame(False)
    ax.set_xlabel("Residual / eV", size=20)
    ax.set_ylabel("Count", size=20)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=16, pad=10, length=7)
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=16, pad=10, length=3)
    tickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(tickmin)
    ax.yaxis.set_minor_locator(tickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()


    return


def scatter(df, xname, yname, bins_col, bins, labels=None):

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(1, 1)

    df1 = df[[xname, yname, bins_col]]
    bins_name = bins_col + "Bin"
    df1[bins_name] = pd.cut(df1[bins_col], bins=bins, labels=labels)

    r = df1[xname].corr(df1[yname], method='pearson')
    rho = df1[xname].corr(df1[yname], method='spearman')

    groups = df1.groupby([bins_name], group_keys=False)

    # Linear Regression
    df_clean = df1.dropna(subset=[xname, yname])
    x = df_clean[xname].values.reshape(-1,1)
    y = df_clean[yname]
    regr = LinearRegression()
    regr.fit(x, y)
    y_hat = regr.predict(x)
    rmse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    m = regr.coef_
    b = regr.intercept_
    residuals = y - y_hat
    p = normaltest(residuals)[1]

    ax = plt.subplot(gs[0])
    dfs = []
    for name, group in groups:
        ax.scatter(group[xname], group[yname], label=name, s=60)
        ax.legend()
        dfs.append(group)

    fit_x = np.linspace(np.min(ax.get_xlim()), np.max(ax.get_xlim()), 100).reshape(-1,1)
    fit_y = regr.predict(fit_x)
    ax.plot(fit_x, fit_y, color='k', lw=1.5, ls='--')

    ax.legend(title="%s" % bins_col).draw_frame(False)

    if "321" in xname:
        xname = xname.replace("321", "B3LYP/3-21G*")
    if "631" in yname:
        yname = yname.replace("631", "B3LYP/6-31G*")

    xtit = xname.split("_")
    xtit = ' '.join([ xtit[0], xtit[-1].upper() ])
    ytit = yname.split("_")
    ytit = ' '.join([ ytit[0], ytit[-1].upper() ])
    ax.set_xlabel("%s / eV" % xtit, size=20)
    ax.set_ylabel("%s / eV" % ytit, size=20)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=16, pad=10, length=7)
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=16, pad=10, length=3)
    tickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(tickmin)
    ax.yaxis.set_minor_locator(tickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()

    # ax.annotate(u'$r$ = %.2f' % r, xy=(0.25,0.90), xycoords='axes fraction', size=24)
    # ax.annotate(u'$\\rho$ = %.2f' % rho, xy=(0.25,0.80), xycoords='axes fraction', size=24)
    ax.annotate(u'$R^2$ = %.2f' % r2, xy=(0.25,0.90), xycoords='axes fraction', size=24)
    # ax.annotate(u'$p$ = %.3e' % p, xy=(0.25,0.60), xycoords='axes fraction', size=24)

    ax.annotate(u'$m$ = %.4f' % m, xy=(0.65,0.15), xycoords='axes fraction', size=24)
    ax.annotate(u'$b$ = %.4f' % b, xy=(0.65,0.05), xycoords='axes fraction', size=24)
    # ax.annotate(u'$RMSE$ = %.4f' % rmse, xy=(0.45,0.05), xycoords='axes fraction', size=24)

    ax.set_aspect('equal')

    # ax = plt.subplot(gs[1])
    # nbins = int(np.log2(len(residuals)) + 1)
    # ax.hist(residuals, bins=nbins)
    # plt.tight_layout()

    return


def dist(df, integral=False):

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(1, 1)

    names = df.columns
    colors = [ "red", "blue", "orange", "green", "yellow", "purple" ]
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

    if integral:
        ax1.set_ylim(0)
        ax1.set_ylabel("Integral", size=20)
        ax1.tick_params(axis='y', which='major', direction='in', labelsize=16, pad=10, length=7)
        ax1.tick_params(axis='both', which='minor', direction='in', labelsize=16, pad=10, length=3)
        ax1.yaxis.set_ticks_position('right')
        ax1.minorticks_on()

    return


def dist_2D(df, bins=None):

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(1, 2, width_ratios=[15,1])

    names = df.columns
    ax = plt.subplot(gs[0])

    sc = ax.hexbin(df.iloc[:,0], df.iloc[:,1], gridsize=len(bins), cmap="Blues")

    ax.set_xlabel("%s" % df.columns[0], size=20)
    ax.set_ylabel("%s" % df.columns[1], size=20)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=16, pad=10, length=7)
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=16, pad=10, length=3)
    tickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(tickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()

    cbax = plt.subplot(gs[-1])
    cb = plt.colorbar(sc, cax=cbax)
    cb.set_label(u"Count", size=24, rotation="vertical", labelpad=10)
    cb.ax.tick_params(labelsize=16, pad=4, direction='in')


    return


if __name__ == '__main__':

    df = pd.read_csv("calibs.csv", index_col=0)

    bins = np.arange(0,101,10)
    labels = [ "0-10", "10-20", "20-30", "30-40", "40-50", "50-60",
              "60-70", "70-80", "80-90", "90-100" ]


    df["GAP_qm_631"] = df["LUMO_qm_631"] - df["HOMO_qm_631"]
    df["HOMO_pm7_cal"] = df["HOMO_pm7"] * 0.8048 + 1.3161
    df["LUMO_pm7_cal"] = df["LUMO_pm7"] * 1.0188 - 0.4888
    df["GAP_pm7_cal"] = df["LUMO_pm7_cal"] - df["HOMO_pm7_cal"]
    df["GAP_pm7_cal"] = df["GAP_pm7_cal"].clip(lower=0)
    df["HOMO_321_cal"] = df["HOMO_qm_321"] * 0.9813 - 0.1908
    df["LUMO_321_cal"] = df["LUMO_qm_321"] * 0.9391 - 0.1695
    df["GAP_321_cal"] = df["LUMO_321_cal"] - df["HOMO_321_cal"]
    df["GAP_321_cal"] = df["GAP_321_cal"].clip(lower=0)

    props = ["HOMO", "LUMO"]
    ref = "qm_631"
    methods = ["pm7", "qm_321"]

    for method in methods:
        for prop in props:

            xname = "%s_%s" % (prop, method)
            yname = "%s_%s" % (prop, ref)
            bins_col = "NAts"

            # Plot data
            scatter(df, xname, yname, bins_col, bins, labels)
            # plt.savefig("%s_%s_%s.svg" % ( prop, method, ref))

            # Plot errors
            residuals(df, xname, yname)
            # plt.savefig("%s_%s_%s_errs.svg" % ( prop, method, ref))

            plt.show()

        # residuals(df, "GAP_qm_631", "GAP_pm7_cal")
        # plt.savefig("GAP_631_pm7_errs.svg")

        # residuals(df, "GAP_qm_631", "GAP_321_cal")
        # plt.savefig("GAP_631_321_errs.svg")
