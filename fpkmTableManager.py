import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
from numpy import *


def read_in_data(cufflinks_t):
    """Function accepts tab delimited FPKM table generated via Cufflinks and returns dictionary

    Args:
        cufflinks_t (str/path): Tab delimited FPKM table.

    Returns:
        Sample dictionary associating event(key) to ordered list(value)

    """
    samples = {}
    for line in open(cufflinks_t):
        if line.startswith('#Gene'):
            header = line.strip().split()
        else:
            line = line.strip().split()
            event = line[0]
            if event not in samples:
                samples[event] = line[1:]
    return samples


def read_in_all_data(heart, brain, quad):
    """Function accepts tab delimited FPKM table generated via Cufflinks and returns dictionaries

    Args:
       heart (str/path): Tissue specific tab delimited FPKM table.
       brain (str/path): Tissue specific tab delimited FPKM table.
       quad (str/path): Tissue specific tab delimited FPKM table.

    Returns:
       Sample dictionary associating event(key) to ordered list(value) for each tissue type

    """
    heart_fpkm = {}
    brain_fpkm = {}
    quad_fpkm = {}
    for line in open(heart):
        if line.startswith('#Gene'):
            continue
        else:
            line = line.strip().split()
            event = line[0]
            if event not in heart_fpkm:
                heart_fpkm[event] = line[1:]
    for line in open(brain):
        if line.startswith('#Gene'):
            continue
        else:
            line = line.strip().split()
            event = line[0]
            if event not in brain_fpkm:
                brain_fpkm[event] = line[1:]
    for line in open(quad):
        if line.startswith('#Gene'):
            continue
        else:
            line = line.strip().split()
            event = line[0]
            if event not in quad_fpkm:
                quad_fpkm[event] = line[1:]
    return heart_fpkm, brain_fpkm, quad_fpkm


def array_all(heart_fpkm, brain_fpkm, quad_fpkm):
    """Function accepts Sample dictionary associating event(key) to ordered list(value) for each tissue type and returns arrays

    Args:
        heart_fpkm (dict): Tissue specific dictionary.
        brain_fpkm (dict): Tissue specific dictionary.
        quad_fpkm (dict): Tissue specific dictionary.

    Returns:
        Float array for tissue specific dictionaries

    """
    quad_mat = []
    heart_mat = []
    brain_mat = []
    keys = heart_fpkm.keys()

    for k in keys:
        quad_mat.append(quad_fpkm[k])
        heart_mat.append(heart_fpkm[k])
        brain_mat.append(brain_fpkm[k])

    quad_array = array(quad_mat).astype(float)
    heart_array = array(heart_mat).astype(float)
    brain_array = array(brain_mat).astype(float)

    return heart_array, brain_array, quad_array


def filterevents(samples, mean_dif, fc):
    """Filter tissue specific dictionary by absolute mean difference between KO and Control as well as abs fold change (KO/WT)

    Args:
        samples (dict): Tissue specific dictionary.
        mean_dif (float): Mean difference metric
        fc (float): Fold change metric

    Returns:
        filtered dictionary where events meet both mean_dic and fc metrics.

    """
    mean_dif = float(mean_dif)
    fc = float(fc)
    filtered_samples = {}
    for k in samples:
        vals = array(samples[k]).astype(float)
        if abs(vals[-2:].mean() - vals[:2].mean()) >= mean_dif:
            if abs(log2(vals[-2:].mean() / vals[:2].mean())) >= fc:
                filtered_samples[k] = samples[k]
    return filtered_samples


def returnfilterfc(cufflinks_t):
    """Function accepts tab delimited FPKM table generated via Cufflinks and returns dictionary containing fold change
    values and the samples the fold change values are derived from.

    Args:
        cufflinks_t (str/path): Tab delimited FPKM table.

    Returns:
        Filtered dictionary containing log10 fold change values between cufflinks_t[-2:]/cufflinks_t[:3]

    """
    samples = read_in_data(cufflinks_t)
    fc = []
    for k in samples:
        vals = array(samples[k]).astype(float)
        y = log10(vals[-2:].mean()) - log10(vals[:3].mean())
        fc.append(y)
    fc = array(fc)
    fc_filt = fc[numpy.logical_not(numpy.isnan(fc))]
    fc_filt = fc_filt[numpy.logical_not(numpy.isinf(fc_filt))]
    return fc_filt, samples


def find_regulated_direction(filtered_samples, fc):
    """Identifies and returns dictionaries containing events that are upregulated or downregulated compared to controls, respectively

    Args:
        filtered_samples (dict): Tissue specific dictionary that has been filtered by fold change and mean difference between groups.
        fc (float): Fold change metric

    Returns:
        Dictionaries where events are either up regulated or down regulated compared to control groups

    """
    upregulated = {}
    downregulated = {}
    keys = filtered_samples.keys()
    for k in keys:
        val = array(filtered_samples[k]).astype(float)
        fold = log10(val[-2:].mean()) - log10(val[:3].mean())
        position = .5 * (log10(val[-2:].mean()) + log10(val[:3].mean()))
        if abs(fold) > fc and ~isnan(fold) and ~isinf(fold) and ~isnan(position) and ~isinf(position):
            if fold > 0:
                upregulated[k] = filtered_samples[k]
            else:
                downregulated[k] = filtered_samples[k]
    return upregulated, downregulated


def plot_pandas_barplot(heart_fpkm, brain_fpkm, quad_fpkm, target, out):
    """Generates a Seaborn barplot depicting the change in expression for a specific gene across tissue type data sets

    Args:
        heart_fpkm (dict): Tissue specific dictionary.
        brain_fpkm (dict): Tissue specific dictionary.
        quad_fpkm (dict): Tissue specific dictionary.
        target (str): Gene of interest
        out (str): Title for saved .eps image

    Returns:
        Nothing. Saves .eps barplot image as out.eps

    """
    heart = array(heart_fpkm[target]).astype(float)[[0, 1, 3, 4, 5, 6, 7, 8]]
    brain = array(brain_fpkm[target]).astype(float)[[0, 1, 3, 4, 5, 6, 7, 8]]
    quad = array(quad_fpkm[target]).astype(float)
    cat = array(['WT', 'WT', 'Saline', 'Saline', 'ASO', 'ASO', 'KO', 'KO'])
    df = pd.DataFrame({'Brain': brain, 'Heart': heart, 'Quad': quad, 'Type': cat})
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    sns.barplot(x='Type', y='Brain', data=df, ax=axes[0]).set(xlabel='', ylabel='Brain')
    sns.barplot(x='Type', y='Heart', data=df, ax=axes[1]).set(xlabel='', ylabel='Heart')
    sns.barplot(x='Type', y='Quad', data=df, ax=axes[2]).set(ylabel='Quad')
    plt.suptitle('$\mu$ FPKM across sample types')
    plt.savefig(out, format='eps', dpi=1000)


def checkfoldchange(tissue_fpkm, out):
    """Generates a histogram depicting the distribution of fold change values between KO and Control. Histogram depicts inner quartile with dashed red line

    Args:
        tissue_fpkm (dict): Tissue specific dictionary.
        out (str): Title for saved .eps image

    Returns:
        Nothing. Saves .eps histogram image as out.eps

    """
    import numpy
    plt.clf()
    fc = []

    for k in tissue_fpkm:
        vals = array(tissue_fpkm[k]).astype(float)
        fc.append(log2(vals[-2:].mean() / vals[:2].mean()))

    fc = array(fc)
    fc_filt = fc[numpy.logical_not(numpy.isnan(fc))]
    fc_filt = fc_filt[numpy.logical_not(numpy.isinf(fc_filt))]

    sns.distplot(fc_filt, color='grey', norm_hist=False, kde=False, bins=1000, vertical=True)
    plt.axhline(y=percentile(fc_filt, [25, 75])[0], linewidth=.8, color='red', linestyle='dashed')
    plt.axhline(y=percentile(fc_filt, [25, 75])[1], linewidth=.8, color='red', linestyle='dashed')
    plt.xlabel('Count')
    plt.ylabel('$\log2$ (KO/WT)')
    red_patch = mpatches.Patch(color='red', label='Inner Quartile\n%s - %s\nn = %s' % (
        round(percentile(fc_filt, [25, 75])[0], 2), round(percentile(fc_filt, [25, 75])[1], 2), int(.5 * len(fc_filt))))
    plt.legend(handles=[red_patch])
    plt.title('KO Fold change distribution')
    plt.savefig(out, format='eps', dpi=1000)


def plotfoldchangeone(cufflinks_t, out):
    """Function accepts tab delimited FPKM table generated via Cufflinks and generates a Seaborn distribution plot identifying genes that fall outside +/- 1 Foldchange

    Args:
        cufflinks_t (str/path): Tab delimited FPKM table.
        out (str): Title for saved .eps image

    Returns:
        Nothing. Saves .eps histogram image as out.eps
    """
    plt.clf()
    fc_filt, samples = returnfilterfc(cufflinks_t)
    sns.distplot(fc_filt, color='grey', norm_hist=False, kde=False, bins=1000, vertical=True)
    plt.axhline(y=1, linewidth=.8, color='red', linestyle='dashed')
    plt.axhline(y=-1, linewidth=.8, color='red', linestyle='dashed')
    plt.xlabel('Count')
    plt.ylabel('$\log2$ (KO/WT)')
    x = [1 for y in fc_filt if 1 >= y >= -1]
    red_patch = mpatches.Patch(color='red', label='+/- 1 Fold change\nn = %s' % (int(len(x))))
    plt.legend(handles=[red_patch])
    plt.title('KO fold change distribution')
    plt.savefig(out, format='eps', dpi=1000)


def plotboxplots(cufflinks_t, target, out):
    """Function accepts tab delimited FPKM table generated via Cufflinks and generates a Seaborn box plot identifying up and downregulated genes as well as marking where
    the gene of interest falls among fold change distributions

    Args:
        cufflinks_t (str/path): Tab delimited FPKM table.
        target (str): Gene of interest to compare fold change distributions to
        out (str): Title for saved .eps image

    Returns:
        Nothing. Saves .eps boxplot image as out.eps

    """
    fc_filt, samples = returnfilterfc(cufflinks_t)
    dmpk = array(samples['Dmpk']).astype(float)
    fc = log10(dmpk[-2:].mean()) - log10(dmpk[:3].mean())
    neg_filt = fc_filt[fc_filt <= -1]
    pos_filt = fc_filt[fc_filt >= 1]
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    if fc >= 0:
        symb = ax2
    else:
        symb = ax1
    ax1.set_xlabel('Negative Fold change')
    ax2.set_xlabel('Positive Fold change')
    sns.boxplot(neg_filt, showfliers=False, color='Orange', notch=True, orient='v', ax=ax1)
    sns.boxplot(pos_filt, showfliers=False, color='Grey', notch=True, orient='v', ax=ax2)
    symb.axhline(y=fc, linewidth=.8, color='red', linestyle='dashed')
    red_patch = mpatches.Patch(color='red', label='DMPK \nFold change = %s' % (round(fc, 2)))
    plt.legend(handles=[red_patch], loc=(.09, .01))
    plt.suptitle('Fold Change Distributions')
    plt.savefig(out, format='eps', dpi=1000)


def filterallevents(heart_fpkm, quad_fpkm, brain_fpkm, fc):
    """Returns set of filtered tissue specific dictionaries based on fold change metric

    Args:
        heart_fpkm (dict): Tissue specific dictionary.
        quad_fpkm (dict): Tissue specific dictionary.
        brain_fpkm (dict): Tissue specific dictionary.
        fc (float): fold change value metric to filter genes be

    Returns:
        Filtered tissue specific dictionaries as well as a set that contains all targets that meet fold change filter criteria across
        all tissue types

    """
    heart_filt = {}
    quad_filt = {}
    brain_filt = {}
    for k in heart_fpkm:
        vals = array(heart_fpkm[k]).astype(float)
        if (vals > 0).sum() > 8:
            if abs(log2(vals[-2:].mean()) - log2(vals[:3].mean())) >= fc:
                heart_filt[k] = heart_fpkm[k]
        else:
            continue
    for k in quad_fpkm:
        vals = array(quad_fpkm[k]).astype(float)
        if (vals > 0).sum() > 7:
            if abs(log2(vals[-2:].mean()) - log2(vals[:3].mean())) >= fc:
                quad_filt[k] = quad_fpkm[k]
        else:
            continue
    for k in brain_fpkm:
        vals = array(brain_fpkm[k]).astype(float)
        if (vals > 0).sum() > 8:
            if abs(log2(vals[-2:].mean()) - log2(vals[:3].mean())) >= fc:
                brain_filt[k] = brain_fpkm[k]
        else:
            continue

    brain_targ = set(brain_filt.keys())
    heart_targ = set(heart_filt.keys())
    quad_targ = set(quad_filt.keys())
    tissue_targs = set.intersection(brain_targ, heart_targ, quad_targ)
    return heart_filt, brain_filt, quad_filt, tissue_targs


def read_in_filtered(heart, quad, brain, fc):
    """Returns set of filtered tissue specific dictionaries based on fold change metric

    Args:
        heart (str/path): Tissue specific cufflinks tab delimited table
        quad (str/path): Tissue specific cufflinks tab delimited table
        brain (str/path): Tissue specific cufflinks tab delimited table
        fc (float): fold change value metric to filture genes be

    Returns:
        Filtered tissue specific dictionaries as well as a set that contains all targets that meet fold change filter criteria across
        all tissue types

    """
    from fpkmTableManager import read_in_all_data as rd
    from fpkmTableManager import filterallevents as fcf
    heart_fpkm, brain_fpkm, quad_fpkm = rd(heart, brain, quad)
    heart_filt, brain_filt, quad_filt, tissue_targs = fcf(heart_fpkm, brain_fpkm, quad_fpkm, fc)
    return heart_filt, brain_filt, quad_filt, tissue_targs


def cluster(filtered_samples, out, fc, brain=False, heart=False, quad=False, metric='euclidean'):
    """Generates a tissue specific Seaborn clustermap based on row Z-scores and designated clustering metric

    Args:
        filtered_samples (dict): Tissue specific dictionary, filtered by both fold change and mean difference metric
        out (str): The title of the Seaborn clustermap generated out+'_<Tissue>.eps'
        fc (float): The fold change metric used to select the events in the filtered_samples dictionary
        brain (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        heart (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        quad (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        metric (str): The Scipy clustering metric in which to order rows and columns of array


    Returns:
        Nothing.

    """
    mat = []
    ylab = []
    for k in filtered_samples:
        ylab.append(k)
        mat.append(filtered_samples[k])

    mat = array(mat).astype(float)
    ylab = array(ylab)
    wt = ["#3498db"]
    het = ["#e74c3c"]
    aso = ["#2ecc71"]
    ko = ["#9b59b6"]

    color_leg = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    legend_lab = ['WT', 'Het', 'ASO', 'KO']
    if quad:
        season_colors = (sns.color_palette(wt, 2) +
                         sns.color_palette(het, 2) +
                         sns.color_palette(aso, 2) +
                         sns.color_palette(ko, 2))
    else:
        season_colors = (sns.color_palette(wt, 3) +
                         sns.color_palette(het, 2) +
                         sns.color_palette(aso, 2) +
                         sns.color_palette(ko, 2))
    if brain:
        title = 'Brain'
        for line in open('TMM_normed_brain_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    if heart:
        title = 'Heart'
        for line in open('TMM_normed_heart_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    if quad:
        title = 'Quad'
        for line in open('TMM_normed_quad_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    sns.set(font_scale=.5)
    g = sns.clustermap(mat, z_score=0, metric=metric, col_colors=season_colors, col_cluster=False,
                       xticklabels=xlab, yticklabels=ylab)
    for C, L in zip([c for c in color_leg], legend_lab):
        g.ax_col_dendrogram.bar(0, 0, color=C, label=L, linewidth=0)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=3)
    plt.suptitle('%s Normalized Fold-change(KO/WT): +/- %s FC' % (title, fc))
    plt.savefig(out+'_%s.pdf' % (title), format='pdf', dpi=1000)


def cluster_all_mat(tissue_targs, heart_fpkm, brain_fpkm, quad_fpkm):
    """Returns fold change normalized array for the mean of the control group across tissue specific dictionaries

    Args:
        tissue_targs (set): Targets identifed to be differentially expressed across all tissues at a set fold change metric
        heart_fpkm (dict): Tissue specific dictionary.
        quad_fpkm (dict): Tissue specific dictionary.
        brain_fpkm (dict): Tissue specific dictionary.

    Returns:
        Fold change normalized matrix containing fold change values for all three tissue types where the fold change for a given sample
        is tissue_fpkm[i]/tissue_fpkm[controls].mean()
        ylab : list containing gene IDs for cluster_all_map function

    """
    mat = []
    ylab = []
    for k in tissue_targs:
        vals = heart_fpkm[k] + brain_fpkm[k] + quad_fpkm[k]
        mat.append(vals)
        ylab.append(k)
    mat = array(mat).astype(float)
    fc_mat = []
    for i in range(mat.shape[1]):
        if i <= 8:
            print 'less than eq 8', i
            fc = list(log2(mat[:, i]) - log2(mat[:, :3].mean(axis=1)))
            fc_mat.append(fc)
        elif 17 >= i > 8:
            print 'greater than 9 less than eq 17', i
            fc = list(log2(mat[:, i]) - log2(mat[:, 9:12].mean(axis=1)))
            fc_mat.append(fc)
        else:
            print 'greater than 17', i
            fc = list(log2(mat[:, i]) - log2(mat[:, 18:21].mean(axis=1)))
            fc_mat.append(fc)
    fc_mat = array(fc_mat).astype(float).transpose()
    return mat, ylab


def cluster_all_map(mat, ylab, out):
    """Returns nothing. Generates a figure where all tissues are clustered together based a Scipy clustering metric provided.

    Args:
        mat (array): Fold change normalized matrix containing fold change values for all three tissue types where the fold change for a given sample
        is tissue_fpkm[i]/tissue_fpkm[controls].mean()
        ylab (list): Y axis labels generated
        out (str): The title of the Seaborn clustermap generated out.pdf'

    Returns:
        Fold change normalized clustermap containing fold change values for all three tissue types where the fold change for a given sample
        is tissue_fpkm[i]/tissue_fpkm[controls].mean(). The Seaborn clustermap will be labeled as <out>.pdf

    """
    heart = ["#3498db"]
    brain = ["#e74c3c"]
    quad = ["#2ecc71"]
    wt = ["#8FBC8F"]
    het = ["#B22222"]
    aso = ["#FFD700"]
    ko = ["#FF69B4"]
    white = ['#FFFFFF']

    color_leg = ["#3498db", "#e74c3c", "#2ecc71", "#FFFFFF", "#8FBC8F", "#B22222", "#FFD700", "#FF69B4"]
    legend_lab = ["heart", "brain", "quad", " ", "wt", "het", "aso", "ko"]
    xlabel = array(["wt", "wt", "wt", "Het", "Het", "aso", "aso", "ko", "ko",
                    "wt", "wt", "wt", "Het", "Het", "aso", "aso", "ko", "ko",
                    "wt", "wt", "Het", "Het", "aso", "aso", "ko", "ko", ])

    sample_type = (sns.color_palette(wt, 3) +
                   sns.color_palette(het, 2) +
                   sns.color_palette(aso, 2) +
                   sns.color_palette(ko, 2) +
                   sns.color_palette(wt, 3) +
                   sns.color_palette(het, 2) +
                   sns.color_palette(aso, 2) +
                   sns.color_palette(ko, 2) +
                   sns.color_palette(wt, 2) +
                   sns.color_palette(het, 2) +
                   sns.color_palette(aso, 2) +
                   sns.color_palette(ko, 2))

    season_colors = (sns.color_palette(heart, 9) +
                     sns.color_palette(brain, 9) +
                     sns.color_palette(quad, 8))

    g = sns.clustermap(mat, annot=False, method='weighted', metric='euclidean', col_colors=[sample_type, season_colors],
                       col_cluster=True, xticklabels=xlabel, yticklabels=ylab)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    for C, L in zip([c for c in color_leg], legend_lab):
        g.ax_col_dendrogram.bar(0, 0, color=C, label=L, linewidth=0)
    g.ax_col_dendrogram.legend(loc="upper right", ncol=2)
    plt.suptitle('Fold change filter across Tissues +/- 1 FC')
    plt.savefig(out, format="pdf", dpi=1000)


def maplot_log2(heart_array, quad_array, brain_array, fc, out):
    """Function accepts tissue specific arrays and generates a non-normalized MA-plot based on a fold change threshold

    Args:
        heart_array (array): Tissue specific array
        quad_array (array): Tissue specific array
        brain_array (array): Tissue specific array
        fc (float): The fold change metric used to select the events in the filtered_samples dictionary
        out (str): The title of the Seaborn regplot <out>.pdf'

    Returns:
        Nothing. Generates a three paneled figure in which the M and A values for each tissue are plotted against each other for a given event.

    """
    fc = float(fc)
    y = log2(heart_array[:, -2:].mean(axis=1)) - log2(heart_array[:, :3].mean(axis=1))
    x = .5 * (log2(heart_array[:, -2:].mean(axis=1)) + log2(heart_array[:, :3].mean(axis=1)))
    z = column_stack((x, y))
    z = z[~isinf(z).any(axis=1)]
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    sns.regplot(z[:, 0], z[:, -1], ci=None, ax=axes[0], fit_reg=False,
                scatter_kws={'alpha': 0.3, 's': 0.5, 'rasterized': False, 'zorder': 1})
    filt = z[(abs(z[:, 1]) > fc)]
    filt = filt[~isnan(filt).any(axis=1)]
    sns.regplot(filt[:, 0], filt[:, -1], fit_reg=False, color='red', ax=axes[0],
                scatter_kws={'alpha': 0.6, 's': 0.7, 'rasterized': False, 'zorder': 1}).set(
        xlabel=' $\mu$ TPM  +/- %s FC: %s genes upregulated  &  %s genes downregulated' % (
        fc, sum(filt[:, 1] > 0), sum(filt[:, 1] < 0)), ylabel='Heart')

    y = log2(quad_array[:, -2:].mean(axis=1)) - log2(quad_array[:, :2].mean(axis=1))
    x = .5 * (log2(quad_array[:, -2:].mean(axis=1)) + log2(quad_array[:, :2].mean(axis=1)))
    z = column_stack((x, y))
    z = z[~isinf(z).any(axis=1)]
    sns.regplot(z[:, 0], z[:, -1], ci=None, ax=axes[1], fit_reg=False,
                scatter_kws={'alpha': 0.3, 's': 0.5, 'rasterized': False, 'zorder': 1})
    filt = z[(abs(z[:, 1]) > fc)]
    filt = filt[~isnan(filt).any(axis=1)]
    sns.regplot(filt[:, 0], filt[:, -1], fit_reg=False, color='red', ax=axes[1],
                scatter_kws={'alpha': 0.6, 's': 0.7, 'rasterized': False, 'zorder': 1}).set(
        xlabel=' $\mu$ TPM  +/- %s FC: %s genes upregulated  &  %s genes downregulated' % (
        fc, sum(filt[:, 1] > 0), sum(filt[:, 1] < 0)), ylabel='Quad')

    y = log2(brain_array[:, -2:].mean(axis=1)) - log2(brain_array[:, :3].mean(axis=1))
    x = .5 * (log2(brain_array[:, -2:].mean(axis=1)) * log2(brain_array[:, :3].mean(axis=1)))
    z = column_stack((x, y))
    z = z[~isinf(z).any(axis=1)]
    sns.regplot(z[:, 0], z[:, -1], ci=None, color='blue', fit_reg=False, ax=axes[2],
                scatter_kws={'alpha': 0.3, 's': 0.5, 'rasterized': False, 'zorder': 1})
    filt = z[(abs(z[:, 1]) > fc)]
    filt = filt[~isnan(filt).any(axis=1)]
    sns.regplot(filt[:, 0], filt[:, -1], fit_reg=False, color='red', ax=axes[2],
                scatter_kws={'alpha': 0.6, 's': 0.7, 'rasterized': False, 'zorder': 1}).set(
        xlabel=' $\mu$ TPM  +/- %s FC: %s genes upregulated  &  %s genes downregulated' % (
        fc, sum(filt[:, 1] > 0), sum(filt[:, 1] < 0)), ylabel='Brain')
    plt.suptitle('Normalized Fold-change(KO/WT) across sample types')
    plt.savefig(out, format='pdf')


def maplot_zerofiltered_log2(heart_array, quad_array, brain_array, fc, out):
    """Function accepts tissue specific arrays and generates a non-normalized MA-plot based on a fold change threshold and only events with non-zero values

    Args:
        heart_array (array): Tissue specific array
        quad_array (array): Tissue specific array
        brain_array (array): Tissue specific array
        fc (float): The fold change metric used to select the events in the filtered_samples dictionary
        out (str): The title of the Seaborn regplot <out>.pdf'

    Returns:
        Nothing. Generates a three paneled figure in which the M and A values for each tissue are plotted against each other for a given event.

    """

    import numpy as np
    fc = float(fc)
    y = log2(heart_array[:, -2:].mean(axis=1)) - log2(heart_array[:, :3].mean(axis=1))
    x = .5 * (log2(heart_array[:, -2:].mean(axis=1)) + log2(heart_array[:, :3].mean(axis=1)))
    z = column_stack((x, y))
    z = z[~isinf(z).any(axis=1)]
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    sns.regplot(z[:, 0], z[:, -1], ci=None, ax=axes[0], fit_reg=False,
                scatter_kws={'alpha': 0.5, 's': 0.5, 'rasterized': False, 'zorder': 1})

    zfilt = heart_array[np.all(heart_array != 0, axis=1)]
    y = log2(zfilt[:, -2:].mean(axis=1)) - log2(zfilt[:, :3].mean(axis=1))
    x = .5 * (log2(zfilt[:, -2:].mean(axis=1)) + log2(zfilt[:, :3].mean(axis=1)))
    z = column_stack((x, y))
    z = z[~isinf(z).any(axis=1)]
    filt = z[(abs(z[:, 1]) >= fc)]
    filt = filt[~isnan(filt).any(axis=1)]

    sns.regplot(filt[:, 0], filt[:, -1], fit_reg=False, color='red', ax=axes[0],
                scatter_kws={'alpha': 0.9, 's': 1.5, 'rasterized': False, 'zorder': 1}).set(
        xlabel=' $\mu$ TPM  +/- %s FC: %s genes upregulated  &  %s genes downregulated' % (
        fc, sum(filt[:, 1] > 0), sum(filt[:, 1] < 0)), ylabel='Heart')

    y1 = log2(quad_array[:, -2:].mean(axis=1)) - log2(quad_array[:, :2].mean(axis=1))
    x1 = .5 * (log2(quad_array[:, -2:].mean(axis=1)) + log2(quad_array[:, :2].mean(axis=1)))
    z1 = column_stack((x1, y1))
    z1 = z1[~isinf(z1).any(axis=1)]

    sns.regplot(z1[:, 0], z1[:, -1], ci=None, fit_reg=False, ax=axes[1],
                scatter_kws={'alpha': 0.5, 's': 0.5, 'rasterized': False, 'zorder': 1})

    zfilt1 = quad_array[np.all(quad_array != 0, axis=1)]
    y2 = log2(zfilt1[:, -2:].mean(axis=1)) - log2(zfilt1[:, :3].mean(axis=1))
    x2 = .5 * (log2(zfilt1[:, -2:].mean(axis=1)) + log2(zfilt1[:, :3].mean(axis=1)))
    z2 = column_stack((x2, y2))
    z2 = z2[~isinf(z2).any(axis=1)]
    filt1 = z2[(abs(z2[:, 1]) >= fc)]
    filt1 = filt1[~isnan(filt1).any(axis=1)]

    sns.regplot(filt1[:, 0], filt1[:, -1], fit_reg=False, color='red', ax=axes[1],
                scatter_kws={'alpha': 0.9, 's': 1.5, 'rasterized': False, 'zorder': 1}).set(
        xlabel=' $\mu$ TPM  +/- %s FC: %s genes upregulated  &  %s genes downregulated' % (
        fc, sum(filt1[:, 1] > 0), sum(filt1[:, 1] < 0)), ylabel='Quad')

    y3 = log2(brain_array[:, -2:].mean(axis=1)) - log2(brain_array[:, :3].mean(axis=1))
    x3 = .5 * (log2(brain_array[:, -2:].mean(axis=1)) + log2(brain_array[:, :3].mean(axis=1)))
    z3 = column_stack((x3, y3))
    z3 = z3[~isinf(z3).any(axis=1)]

    sns.regplot(z3[:, 0], z3[:, -1], ci=None, color='blue', fit_reg=False, ax=axes[2],
                scatter_kws={'alpha': 0.5, 's': 0.5, 'rasterized': False, 'zorder': 1})

    zfilt2 = brain_array[np.all(brain_array != 0, axis=1)]
    y4 = log2(zfilt2[:, -2:].mean(axis=1)) - log2(zfilt2[:, :3].mean(axis=1))
    x4 = .5 * (log2(zfilt2[:, -2:].mean(axis=1)) + log2(zfilt2[:, :3].mean(axis=1)))
    z4 = column_stack((x4, y4))
    z4 = z4[~isinf(z4).any(axis=1)]
    filt2 = z4[(abs(z4[:, 1]) >= fc)]
    filt2 = filt2[~isnan(filt2).any(axis=1)]

    sns.regplot(filt2[:, 0], filt2[:, -1], fit_reg=False, color='red', ax=axes[2],
                scatter_kws={'alpha': 0.9, 's': 1.5, 'rasterized': False, 'zorder': 1}).set(
        xlabel=' $\mu$ TPM  +/- %s FC: %s genes upregulated  &  %s genes downregulated' % (
        fc, sum(filt2[:, 1] > 0), sum(filt2[:, 1] < 0)), ylabel='Brain')

    plt.suptitle('Normalized Fold-change(KO/WT) across sample types')
    plt.savefig(out, format='pdf')


def getReadCov(bamdir, out_f):
    """Function generates a file listing total number of mapped reads and non-mitochondria reads for a given bam.

    Args:
        bamdir (str/path): Directory containing bam files
        out_f (str): The title of document to be generated containing mapped reads for a given bam

    Returns:
        Text file that indicates the total number of mapped reads and non-mitochondrial reads for a given bam directory.

    """
    out = open(out_f, 'w')
    out.write("#Sample\tTot\tnonRiboMito\n")
    files = [f for f in os.listdir(bamdir) if f.endswith('.bam')]

    for f in files:

        statlines = pysam.idxstats(os.path.join(bamdir, f)).strip().split('\n')

        tot = 0
        nonribomito = 0
        for line in statlines:
            vals = line.strip().split()
            n = int(vals[2])
            tot += n
            if vals[0] not in ['chrRibo', 'chrM']:
                nonribomito += n
        out.write("\t".join(map(str, [f, tot, nonribomito])) + "\n")

    out.close()


def tmm_estimate(ctlname, cts_f, tpm,  out_f, Mtrimfactor= .3, Atrimfactor=.05):
    """Function generates a file listing TMM normalized gene FPKM/TPM values. Re-centers low read coverage genes around zero.

    Args:
        ctlname (str): Control column header that the values will be normalized to
        cts_f (str/path): Table containing total mapped reads as well as non-mitochondrial mapped reads.
        tpm (str/path): Table of FPKM/TPM values to be normalized
        out_f (str): Title of new TMM normalized FPKM/TPM table
        Mtrimfactor (float): Float value to trim M for MA-plot distribution to remove outliers from expression table
        Atrimfactor (float): Float value to trim A for MA-plot distribution to remove outliers from expression table

    Returns:
        Normed text file that represents the TMM normalized values for a given FPKM/TPM expression table

    """
    nummapped = {}
    tpmD = {}
    for line in open(cts_f):
        if not line.startswith("#"):
            f, totmap, nonribomap = line.strip().split()
            nummapped[f.split(".")[0]] = int(nonribomap)
    for line in open(tpm):

        if line.startswith('target_id'):
            files = line.strip().split('\t')[1:]
            for col in range(len(files)):
                if files[col] == ctlname:
                    ctlindex = col
        else:
            line = line.strip().split('\t')
            gene = line[0]
            tpms = line[1:]
            tpmD[gene] = map(float, tpms)
    cts = array([tpmD[g] for g in tpmD])
    print cts
    ctlnum = float(nummapped[ctlname])
    nonzerocts = cts[where(cts[:, ctlindex] > 0)[0], :]
    TMMs = []

    for i in range(len(files)):
        name = files[i]

        if name != ctlname:
            num = float(nummapped[name])
            workingcts = nonzerocts[where(nonzerocts[:, i] > 0)[0], :]
            M = log2((workingcts[:, ctlindex] / ctlnum) / (workingcts[:, i] / num))
            A = .5 * log2((workingcts[:, ctlindex] / ctlnum) * (workingcts[:, i] / num))
            W = 1. / ((ctlnum - workingcts[:, ctlindex]) / (ctlnum * workingcts[:, ctlindex]) + \
                      (num - workingcts[:, i]) / (num * workingcts[:, i]))
            MAW = vstack([M, A, W]).T

            Mtrimfactor = float(Mtrimfactor)
            Mlow = sorted(M)[int(floor(.5 * Mtrimfactor * len(M)))]
            Mhigh = sorted(M)[int(ceil((1 - .5 * Mtrimfactor) * len(M)))]
            Atrimfactor = float(Atrimfactor)
            Alow = sorted(A)[int(floor(.5 * Atrimfactor * len(A)))]
            Ahigh = sorted(A)[int(ceil((1 - .5 * Atrimfactor) * len(A)))]

            MAW = MAW[where(MAW[:, 0] > Mlow)[0], :]
            MAW = MAW[where(MAW[:, 0] < Mhigh)[0], :]
            MAW = MAW[where(MAW[:, 1] > Alow)[0], :]
            MAW = MAW[where(MAW[:, 1] < Ahigh)[0], :]

            logTMM = (MAW[:, 0] * MAW[:, 2]).sum() / MAW[:, 2].sum()
            TMMs.append(logTMM)
            print files[i], logTMM
            plt.plot(A, M, '.', color = '#97A6AA')
            plt.savefig('MAplot.5.03.pdf')
        else:
            print files[i]
            TMMs.append(0)

    TMMs = 2 ** array(TMMs)
    out = open(out_f, 'w')
    genes = tpmD.keys()
    genes.sort()
    out.write('#Gene\t' + "\t".join(f for f in files)+"\n")
    for gene in genes:
        out.write(gene + "\t")
        normed = array(tpmD[gene]) * TMMs
        out.write("\t".join(map(str, map(round, normed, \
                                         [2 for x in range(len(normed))]))) + "\n")

    print tpmD
    return normed
    out.close()


def MAplot(rpkmtable_f, out_f):
    """Function takes a FPKM/TPM table and generates an MA-plot and saves it as out_f.png
    Args:
        rpkmtable_f (str/path): Table of FPKM/TPM values to be normalized
        out_f (str): Title of MA-plot Figure

    Returns:
        Normed text file that represents the TMM normalized values for a given FPKM/TPM expression table

    """
    import sys, os, operator, subprocess, math
    import matplotlib
    import pysam
    import seaborn as sns
    from numpy import *
    from scipy.stats.stats import pearsonr
    from pylab import pcolor, colorbar, xticks, yticks
    from matplotlib import pyplot as plt
    matplotlib.use('Agg')
    data = []
    for line in open(rpkmtable_f):
        if line.startswith("#"):
            samples = line.strip().split()[1:]
        else:
            rpkm = map(float, line.strip().split("\t")[1:])
            data.append(rpkm)
    data = array(data)
    print data.shape[0], "genes"
    print data.shape[1], "samples"

    n = 1
    figure(figsize=(11, 14))
    for i in range(len(samples)):
        for j in range(len(samples)):
            subplot(len(samples), len(samples), n)
            if j < i:
                m = log10(data[:, j]) - log10(data[:, i])
                a = .5 * (log10(data[:, j]) + log10(data[:, i]))
                scatter(a, m, s=.2, zorder=1, alpha=.5, \
                        lw=0, rasterized=True, color='k')
                title(samples[j] + ".vs\n" + samples[i], fontsize=6)
                xlim(-3, 6)
                ylim(-4, 4)
                xticks([-1, 2, 4], [-1, 2, 4], fontsize=6)
                yticks([-4, 0, 4], [-4, 0, 4], fontsize=6)
                ylabel("M", fontsize=6)
                xlabel("A", fontsize=6)
                print i, j
            else:
                xticks([])
                yticks([])
            n += 1
    subplots_adjust(hspace=.8, wspace=.5)
    savefig(out_f, dpi=300)


def cluster_and_correlate_target(tissue_fpkm, tissue_filt, target, coef=0.85, out, metric='euclidean', brain=False, heart=False, quad=False):
    """Generates a tissue specific Seaborn clustermap based on row Z-scores and designated clustering metric

    Args:
        tissue_fpkm (dict): Tissue specific dictionary
        tissue_filt (dict): Tissue specific dictionary, filtered by both fold change and mean difference metric
        target (str): Gene in which the spearman correlation will be against
        coef (float): Absolute spearman correlation coefficient (rho) to filter events by
        out (str): The title of the Seaborn clustermap generated out+'_<Tissue>.eps'
        fc (float): The fold change metric used to select the events in the filtered_samples dictionary
        brain (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        heart (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        quad (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        metric (str): The Scipy clustering metric in which to order rows and columns of array


    Returns:
        Fold change normalized clustermap containing fold change values for specific tissue type where the fold change for a given sample
        is tissue_fpkm[i]/tissue_fpkm[controls].mean(). The Seaborn clustermap will be labeled as <out>_<tissue>.pdf

    """
    from scipy.stats import spearmanr
    cor_dic = {}

    wt = ["#3498db"]
    het = ["#e74c3c"]
    aso = ["#2ecc71"]
    ko = ["#9b59b6"]
    color_leg = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    legend_lab = ['WT', 'Het', 'ASO', 'KO']
    if quad:
        season_colors = (sns.color_palette(wt, 2) +
                         sns.color_palette(het, 2) +
                         sns.color_palette(aso, 2) +
                         sns.color_palette(ko, 2))
    else:
        season_colors = (sns.color_palette(wt, 3) +
                         sns.color_palette(het, 2) +
                         sns.color_palette(aso, 2) +
                         sns.color_palette(ko, 2))
    if brain:
        title = 'Brain'
        for line in open('TMM_normed_brain_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    if heart:
        title = 'Heart'
        for line in open('TMM_normed_heart_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    if quad:
        title = 'Quad'
        for line in open('TMM_normed_quad_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    targ = array(tissue_fpkm[target]).astype(float)
    for k in tissue_fpkm:
        if k != target:
            vals = array(tissue_fpkm[k]).astype(float)
            if abs(spearmanr(targ, vals)[0]) >= coef:
                cor_dic[k] = tissue_fpkm[k]
    fks = set(tissue_filt.keys())
    cks = set(cor_dic.keys())
    tissue_cor = set.intersection(cks, fks)
    mat = []
    ylab = []
    for k in tissue_cor:
        mat.append(tissue_filt[k])
        ylab.append(k)
    mat = array(mat).astype(float)
    ylab = array(ylab)
    g = sns.clustermap(mat, z_score=0, metric=metric, col_colors=season_colors, col_cluster=False,
                       xticklabels=xlab, yticklabels=ylab)
    for C, L in zip([c for c in color_leg], legend_lab):
        g.ax_col_dendrogram.bar(0, 0, color=C, label=L, linewidth=0)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=3)
    plt.suptitle(r'%s- %s TPM $\rho$ >= 0.85 & filtered +/- 2.0 FC' % (title, target)).set_fontsize(fontsize=10.0)
    plt.savefig(out+'_%s.pdf' % (title), format='pdf', dpi=1000)


def foldchange_cluster_w_rho(tissue_fpkm, tissue_filt, target, out, coef=0.65,  brain=False, heart=False, quad=False, metric='euclidian'):
    """Generates a tissue specific Seaborn clustermap based on normalized row Z-scores based on the fold change between mean control and all samples and designated clustering metric

    Args:
        tissue_fpkm (dict): Tissue specific dictionary
        tissue_filt (dict): Tissue specific dictionary, filtered by both fold change and mean difference metric
        target (str): Gene in which the spearman correlation will be against
        coef (float): Absolute spearman correlation coefficient (rho) to filter events by
        out (str): The title of the Seaborn clustermap generated out+'_<Tissue>.eps'
        brain (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        heart (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        quad (bool): Identifies which tissue and column headers to use for clustering and labeling purposes
        metric (str): The Scipy clustering metric in which to order rows and columns of array


    Returns:
        Fold change normalized clustermap containing fold change values for specific tissue type where the fold change for a given sample
        is tissue_fpkm[i]/tissue_fpkm[controls].mean(). The Seaborn clustermap will be labeled as <out>_<tissue>.pdf

    """
    from scipy.stats import spearmanr
    cor_dic = {}
    wt = ["#3498db"]
    het = ["#e74c3c"]
    aso = ["#2ecc71"]
    ko = ["#9b59b6"]
    color_leg = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    legend_lab = ['WT', 'Het', 'ASO', 'KO']
    if quad:
        season_colors = (sns.color_palette(wt, 2) +
                         sns.color_palette(het, 2) +
                         sns.color_palette(aso, 2) +
                         sns.color_palette(ko, 2))
    else:
        season_colors = (sns.color_palette(wt, 3) +
                         sns.color_palette(het, 2) +
                         sns.color_palette(aso, 2) +
                         sns.color_palette(ko, 2))
    if brain:
        idx = 3
        title = 'Brain'
        for line in open('TMM_normed_brain_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    if heart:
        idx = 3
        title = 'Heart'
        for line in open('TMM_normed_heart_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break
    if quad:
        idx = 2
        title = 'Quad'
        for line in open('TMM_normed_quad_dmpkKO.txt'):
            if line.startswith('#Gene'):
                xlab = array(line.strip().split()[1:])
                break
            break

    targ = array(tissue_fpkm[target]).astype(float)
    for k in tissue_fpkm:
        if k != target:
            vals = array(tissue_fpkm[k]).astype(float)
            if abs(spearmanr(targ, vals)[0]) >= coef:
                cor_dic[k] = tissue_fpkm[k]
    fks = set(tissue_filt.keys())
    cks = set(cor_dic.keys())
    tissue_cor = set.intersection(cks, fks)
    mat = []
    ylab = []
    fc_mat = []

    for k in tissue_cor:
        mat.append(tissue_filt[k])
        ylab.append(k)
    mat = array(mat).astype(float)
    ylab = array(ylab)
    for i in range(mat.shape[1]):
        fc = list(log2(mat[:, i]) - log2(mat[:, :idx].mean(axis=1)))
        fc_mat.append(fc)
    fc_mat = array(fc_mat).astype(float).transpose()

    sns.set(font_scale=.5)
    g = sns.clustermap(fc_mat, metric=metric, col_colors=season_colors, col_cluster=False,
                       xticklabels=xlab, yticklabels=ylab)
    for C, L in zip([c for c in color_leg], legend_lab):
        g.ax_col_dendrogram.bar(0, 0, color=C, label=L, linewidth=0)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=3)
    plt.suptitle(r'%s-$\mu$ Normalized Fold-change(KO/WT): +/- 2.0 FC' % (title))
    plt.savefig(out+'_%s.pdf' % (title), format='pdf', dpi=1000)


def volcano_plot(tissue_array, type=None, out):
    """Generates a tissue specific volcano plot based on the log2 fold change (x) and log10(p-values from Tukeys T-test

    Args:
        tissue_array (array): Tissue specific array
        type (str): Tissue data set was derived from; used for labeling plot
        out (str): The title of the Volcano plot generated <Tissue>+'-<out>'.pdf'

    Returns:
        Nothing. Generates a Volcano plot for a tissue specific array

    """
    plt.clf()
    sns.set(font_scale=1.4)
    from scipy.stats import ttest_ind
    filt = tissue_array[np.all(tissue_array != 0, axis=1)]
    x = log2(filt[:, -2:].mean(axis=1)) - log2(filt[:, :3].mean(axis=1))
    y = -log10(ttest_ind(filt[:, -2:], filt[:, :2], axis=1)[1:][0])
    xy = column_stack((x, y))
    xy = xy[~isinf(xy).any(axis=1)]
    sns.regplot(xy[:, 0], xy[:, 1], fit_reg=False, color='k',
                scatter_kws={'alpha': 0.9, 's': 2.0, 'rasterized': False, 'zorder': 1}).set_ylim(0, )
    de = xy[abs(xy[:, 0]) > 1, :]
    de = de[de[:, 1] > 2, :]
    up = sum(de[:, 0] > 0)
    down = sum(de[:, 0] < 0)
    sns.regplot(de[:, 0], de[:, 1], fit_reg=False, color='r',
                scatter_kws={'alpha': 0.9, 's': 2.5, 'rasterized': False, 'zorder': 1})
    plt.axhline(y=2.0, linewidth=.8, color='red', linestyle='dashed')
    plt.axvline(x=1.0, linewidth=.8, color='red', linestyle='dashed')
    plt.axvline(x=-1.0, linewidth=.8, color='red', linestyle='dashed')
    plt.xlabel(r'$\log_2$(KO/WT)')
    plt.ylabel(r'-$\log_{10}$ p-value')
    plt.suptitle('%s: Downregulated genes: %s    Upregulated genes: %s ' % (type, down, up))
    plt.savefig('%s-'+out % (type), format='pdf')