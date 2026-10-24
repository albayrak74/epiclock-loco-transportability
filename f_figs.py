"""Figures 1-4 (American spelling), 600 dpi TIFF + PNG."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, os
from PIL import Image
W = './work/'; FIG = './fig/'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C = ['GSE36054', 'GSE40279', 'GSE41169', 'GSE64495', 'GSE69270']
LAB = {'GSE36054': 'pediatric', 'GSE40279': 'Hannum', 'GSE41169': 'Dutch', 'GSE64495': 'mixed-age', 'GSE69270': 'Young Finns'}
COL = dict(zip(C, ['#2A78D6', '#EB6834', '#1BAF7A', '#F0A30A', '#7E5BA9']))
P = pd.read_csv(W + 'loco_predictions.csv', index_col=0)


def save(fig, n):
    fig.savefig(FIG + f'Figure{n}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG + f'fig{n}.tiff', dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    im = Image.open(FIG + f'fig{n}.tiff'); im.convert('RGB').save(FIG + f'fig{n}.tiff', dpi=(600, 600), compression='tiff_lzw')
    plt.close(fig)


# Figure 1
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={'wspace': 0.32})
for g in C:
    d = P[P.cohort == g]
    ax[0].scatter(d.Age, d.EN, s=6, alpha=0.65, color=COL[g], lw=0,
                  label=f'{g} ({LAB[g]})  MAE {np.mean(np.abs(d.EN - d.Age)):.1f} y')
ax[0].plot([0, 105], [0, 105], 'k--', lw=0.7); ax[0].set_ylim(-3, 112)
ax[0].set_xlabel('Chronological age (years)'); ax[0].set_ylabel('Predicted DNAm age (years)')
ax[0].legend(frameon=False, fontsize=6.0, loc='upper left', bbox_to_anchor=(0.0, 1.0), markerscale=1.5, handletextpad=0.2)
ax[0].set_title('A  LOCO predictions (elastic net)', loc='left', fontsize=9)
d = P[P.cohort == 'GSE40279']; r = d.EN - d.Age
ax[1].scatter(d.Age, r, s=6, alpha=0.6, color=COL['GSE40279'], lw=0)
dec = (d.Age // 10) * 10; mm = r.groupby(dec).agg(['mean', 'size']); mm = mm[mm['size'] > 1]
ax[1].plot(mm.index + 5, mm['mean'], 'k-o', ms=3, lw=1)
tmax = max(P[P.cohort == g].Age.max() for g in C if g != 'GSE40279')
ax[1].axhline(0, color='k', ls='--', lw=0.7); ax[1].axvline(tmax, color='grey', ls=':', lw=0.8)
ax[1].text(tmax + 1, ax[1].get_ylim()[0] + 2, f'oldest training age ({tmax:.1f} y)', fontsize=6.5, color='grey')
ax[1].set_xlabel('Chronological age (years)'); ax[1].set_ylabel('Predicted − chronological age (years)')
ax[1].set_title('B  Residuals in GSE40279', loc='left', fontsize=9)
save(fig, 1)

# Figure 2
T = pd.read_csv(W + 'T2_loco.csv').set_index('cohort')
fig, ax = plt.subplots(figsize=(7.0, 3.2)); x = np.arange(len(C)); w = 0.2
specs = [('EN', 'Elastic net', '#2A78D6'), ('GBM', 'Gradient boosting', '#EB6834'), ('ENS', 'Ensemble', '#1BAF7A'),
         ('SC', 'Single-cohort EN (trained on GSE40279)', '#9CA0A6')]
for j, (m, lab, c) in enumerate(specs):
    v = [T.loc[g, f'{m}_mae'] if f'{m}_mae' in T and not np.isnan(T.loc[g, f'{m}_mae']) else np.nan for g in C]
    err = None
    if m in ('EN', 'GBM'):
        err = np.array([[T.loc[g, f'{m}_mae'] - T.loc[g, f'{m}_mae_lo'] for g in C], [T.loc[g, f'{m}_mae_hi'] - T.loc[g, f'{m}_mae'] for g in C]])
    ax.bar(x + (j - 1.5) * w, v, w, color=c, label=lab, yerr=err, capsize=2, error_kw={'lw': 0.8})
ax.text(1 + 1.5 * w, 0.2, 'n/a', rotation=90, fontsize=7, ha='center')
ax.set_xticks(x); ax.set_xticklabels([f'{g}\n({LAB[g]})' for g in C], fontsize=8)
ax.set_ylabel('MAE on held-out cohort (years)'); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); save(fig, 2)

# Figure 3
D = pd.read_csv(W + 'diversity.csv')
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for g in C:
    d = D[D.target == g].groupby('k').mae.agg(['median', 'min', 'max'])
    ax.plot(d.index, d['median'], '-o', ms=3.5, color=COL[g], label=f'{g} ({LAB[g]})')
    ax.fill_between(d.index, d['min'], d['max'], color=COL[g], alpha=0.12, lw=0)
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel('Number of training cohorts'); ax.set_ylabel('Held-out MAE (years)')
ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); save(fig, 3)

# Figure 4
Fw = pd.read_csv(W + 'fewshot.csv'); Fw = Fw[Fw['mode'] == 'intercept']
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for g in C:
    d = Fw[Fw.cohort == g].sort_values('m')
    unc = P[P.cohort == g]; u = np.mean(np.abs(unc.EN - unc.Age))
    xs = [0] + list(d.m); ys = [u] + list(d.mae)
    ax.plot(xs, ys, '-o', ms=3.5, color=COL[g], label=f'{g} ({LAB[g]})')
    ax.errorbar(d.m, d.mae, yerr=[d.mae - d.lo, d.hi - d.mae], fmt='none', ecolor=COL[g], capsize=2, lw=0.9)
ax.set_xlabel('Labeled target-cohort samples used for offset calibration'); ax.set_ylabel('MAE on remaining samples (years)')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); save(fig, 4)
print('figures done')
