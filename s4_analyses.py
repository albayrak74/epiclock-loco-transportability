"""All secondary analyses on the reproduced LOCO predictions."""
import itertools, sys
from common import *
from scipy.optimize import nnls
import statsmodels.formula.api as smf
P = pd.read_csv(W + 'loco_predictions.csv', index_col=0)
rng = np.random.default_rng(2026)
OUT = {}
mae = lambda p, a: float(np.mean(np.abs(p - a)))


def boot(h, f, B=2000):
    d = P[P.cohort == h]; n = len(d); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); v.append(f(d.iloc[i]))
    return np.percentile(v, [2.5, 97.5])


def calib(d, col='EN'):
    b = np.polyfit(d[col].values, d.Age.values, 1)        # chronological ~ predicted
    return b[0], b[1]


rows = []
for h in C:
    d = P[P.cohort == h]
    r = {'cohort': h, 'n': len(d)}
    for m in ['EN', 'GBM', 'ENS', 'GBM_raw', 'SC']:
        if d[m].isna().all(): continue
        r[f'{m}_mae'] = mae(d[m], d.Age)
    r['EN_medae'] = float(np.median(np.abs(d.EN - d.Age))); r['EN_r'] = float(np.corrcoef(d.EN, d.Age)[0, 1])
    r['EN_bias'] = float((d.EN - d.Age).mean())
    r['EN_mae_lo'], r['EN_mae_hi'] = boot(h, lambda x: mae(x.EN, x.Age))
    r['GBM_mae_lo'], r['GBM_mae_hi'] = boot(h, lambda x: mae(x.GBM, x.Age))
    r['dGBM'] = r['GBM_mae'] - r['EN_mae']
    r['dGBM_lo'], r['dGBM_hi'] = boot(h, lambda x: mae(x.GBM, x.Age) - mae(x.EN, x.Age))
    if not d.SC.isna().all():
        r['dSC'] = r['SC_mae'] - r['EN_mae']
        r['dSC_lo'], r['dSC_hi'] = boot(h, lambda x: mae(x.SC, x.Age) - mae(x.EN, x.Age))
    s, i = calib(d); r['cal_slope'] = s; r['cal_int'] = i
    r['cal_slope_lo'], r['cal_slope_hi'] = boot(h, lambda x: calib(x)[0])
    rows.append(r)
T2 = pd.DataFrame(rows); T2.to_csv(W + 'T2_loco.csv', index=False)
print(T2.round(2).to_string())
allP = P
OUT['pooled_mae'] = mae(allP.EN, allP.Age); OUT['pooled_medae'] = float(np.median(np.abs(allP.EN - allP.Age)))

# ---- age bands, every cohort
bands = [(0, 10, '0–9'), (10, 20, '10–19'), (20, 40, '20–39'), (40, 50, '40–49'), (50, 60, '50–59'), (60, 70, '60–69'),
         (70, 80, '70–79'), (80, 120, '≥80')]
tr_age = {h: np.concatenate([PH[g].Age.values for g in C if g != h]) for h in C}
rows = []
for h in C:
    d = P[P.cohort == h]
    for lo, hi, lab in bands:
        x = d[(d.Age >= lo) & (d.Age < hi)]
        if len(x) < 5: continue
        ta = tr_age[h]
        rows.append({'cohort': h, 'band': lab, 'n': len(x), 'bias': float((x.EN - x.Age).mean()), 'mae': mae(x.EN, x.Age),
                     'n_train': int(((ta >= lo) & (ta < hi)).sum())})
AB = pd.DataFrame(rows); AB.to_csv(W + 'T3_agebands.csv', index=False); print(AB.round(2).to_string())
# GSE40279 main-text bands
d = P[P.cohort == 'GSE40279']; ta = tr_age['GSE40279']
for lab, msk, tm in [('le50', d.Age <= 50, ta <= 50), ('51_74', (d.Age > 50) & (d.Age <= 74), (ta > 50) & (ta <= 74)), ('gt74', d.Age > 74, ta > 74)]:
    OUT[f'b_{lab}_n'] = int(msk.sum()); OUT[f'b_{lab}_bias'] = float((d.EN - d.Age)[msk].mean()); OUT[f'b_{lab}_mae'] = mae(d.EN[msk], d.Age[msk])
    OUT[f'b_{lab}_ntrain'] = int(tm.sum())
OUT['train_gt50'] = int((ta > 50).sum()); OUT['train_gt60'] = int((ta > 60).sum()); OUT['train_max'] = float(ta.max())
OUT['slope_resid_age_40279'] = float(np.polyfit(d.Age, d.EN - d.Age, 1)[0])
json.dump(OUT, open(W + 'V_part1.json', 'w'), indent=1)
print(OUT)
