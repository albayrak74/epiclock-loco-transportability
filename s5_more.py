"""Diversity (75 models), few-shot recalibration, sex residuals, cell composition, published clocks, final clock."""
import itertools, sys
from common import *
from scipy.optimize import nnls
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
P = pd.read_csv(W + 'loco_predictions.csv', index_col=0)
mae = lambda p, a: float(np.mean(np.abs(np.asarray(p) - np.asarray(a))))
what = sys.argv[1:] or ['div', 'few', 'sex', 'cell', 'pub', 'final']

if 'div' in what:
    rows = []
    for h in C:
        oth = [g for g in C if g != h]
        for k in range(1, 5):
            for tr in itertools.combinations(oth, k):
                m = fit_en(list(tr), n_alphas=12, folds=3)
                rows.append({'target': h, 'k': k, 'train': '+'.join(tr), 'mae': mae(pred_en(m, h), PH[h].Age)})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(W + 'diversity.csv', index=False)

if 'few' in what:
    rng = np.random.default_rng(2026); rows = []
    for h in C:
        d = P[P.cohort == h].reset_index(drop=True); n = len(d)
        for m in [5, 10, 20, 40]:
            for mode in ['intercept', 'slope']:
                if mode == 'slope' and m < 10: continue
                e0, e1 = [], []
                for _ in range(200):
                    i = rng.choice(n, m, replace=False); rest = np.setdiff1d(np.arange(n), i)
                    if mode == 'intercept':
                        corr = d.EN[rest] + (d.Age[i] - d.EN[i]).mean()
                    else:
                        b = np.polyfit(d.EN[i], d.Age[i], 1); corr = b[0] * d.EN[rest] + b[1]
                    e0.append(mae(d.EN[rest], d.Age[rest])); e1.append(mae(corr, d.Age[rest]))
                rows.append({'cohort': h, 'm': m, 'mode': mode, 'mae_uncal': np.mean(e0), 'mae': np.mean(e1),
                             'lo': np.percentile(e1, 2.5), 'hi': np.percentile(e1, 97.5)})
    F_ = pd.DataFrame(rows); F_.to_csv(W + 'fewshot.csv', index=False); print(F_.round(2).to_string())

if 'sex' in what:
    rows = []
    for h in C:
        d = P[P.cohort == h].copy()
        for q in [1, 2]:
            b = np.polyfit(d.Age, d.EN, q); d[f'res{q}'] = d.EN - np.polyval(b, d.Age)
        rows.append(d)
    D = pd.concat(rows); D['male'] = (D.Gender.str.lower().str.startswith('m')).astype(int)
    print(D.groupby('cohort').male.agg(['sum', 'count']))
    for q in [1, 2]:
        f = smf.ols(f'res{q} ~ male + cohort', D).fit()
        print('sex', q, f.params['male'], f.bse['male'], f.pvalues['male'], 'df', f.df_resid)
    D[['cohort', 'Age', 'Gender', 'male', 'res1', 'res2']].to_csv(W + 'sex_resid.csv')

if 'cell' in what:
    BL = '/usr/local/lib/python3.13/dist-packages/biolearn/data/'
    ref = pd.read_csv(BL + '450K_reinius_12_reference.csv', index_col=0)
    rows = []
    for h in C:
        X = PANEL[h]; cp = [c for c in ref.index if c in X.columns and X[c].notna().mean() > 0.5]
        Xr = X[cp].fillna(X[cp].mean()); A = ref.loc[cp].values
        props = np.array([nnls(A, Xr.iloc[i].values)[0] for i in range(len(Xr))])
        props = props / props.sum(1, keepdims=True)
        d = P[P.cohort == h].copy(); d[ref.columns] = props
        d['res'] = d.EN - d.Age
        f0 = smf.ols('res ~ Age', d).fit(); f1 = smf.ols('res ~ Age + ' + ' + '.join(ref.columns[1:]), d).fit()
        rows.append({'cohort': h, 'n_cpg': len(cp), **{c: float(d[c].mean()) for c in ref.columns},
                     'slope_unadj': f0.params['Age'], 'slope_adj': f1.params['Age'],
                     'r2_cells': float(smf.ols('res ~ ' + ' + '.join(ref.columns[1:]), d).fit().rsquared)})
        d.to_csv(W + f'cells_{h}.csv')
    CE = pd.DataFrame(rows); CE.to_csv(W + 'cells.csv', index=False); print(CE.round(3).to_string())

if 'pub' in what:
    def anti_trafo(x, adult_age=20):  # as in biolearn.model
        return np.where(x < 0, (1 + adult_age) * np.exp(x) - 1, (1 + adult_age) * x + adult_age)
    BL = '/usr/local/lib/python3.13/dist-packages/biolearn/data/'
    allm = pd.concat([PANEL[g] for g in C])
    defs = {'Horvath multi-tissue': ('Horvath1', lambda s: anti_trafo(s + 0.696)),
            'Horvath skin-and-blood': ('Horvath2', lambda s: anti_trafo(s - 0.447119319)),
            'Hannum': ('Hannum', lambda s: s), 'Lin': ('Lin', lambda s: s)}
    rows = []
    for name, (f, tf) in defs.items():
        co = pd.read_csv(BL + f + '.csv').set_index('CpGmarker').CoefficientTraining
        icpt = co.pop('intercept') if 'intercept' in co.index else (co.pop('Intercept') if 'Intercept' in co.index else 0.0)
        for h in C:
            X = PANEL[h].reindex(columns=co.index)
            cov = float(X.notna().any().mean())
            X = X.fillna(allm.reindex(columns=co.index).mean()).fillna(0)
            p = tf(X.values @ co.values + icpt)
            a = PH[h].Age.values
            rows.append({'clock': name, 'cohort': h, 'coverage': cov, 'mae': mae(p, a), 'bias': float(np.mean(p - a))})
    PB = pd.DataFrame(rows); PB.to_csv(W + 'published.csv', index=False); print(PB.round(2).to_string())

if 'final' in what:
    m = fit_en(C)
    co = pd.Series(m[0].coef_, index=m[1]); nz = co[co != 0]
    print('final nonzero', len(nz), (nz > 0).sum(), (nz < 0).sum(), 'intercept', m[0].intercept_)
    nz.reindex(nz.abs().sort_values(ascending=False).index).to_csv(W + 'final_clock.csv')
    json.dump({'intercept': float(m[0].intercept_), 'alpha': float(m[0].alpha_)}, open(W + 'final_intercept.json', 'w'))
    print(nz.abs().sort_values(ascending=False).head(8))
