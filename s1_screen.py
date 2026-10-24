"""Step 1: stream one MethAgingDB cohort, compute per-CpG Pearson r with age, n, mean, sd.
Usage: python s1_screen.py GSE41169"""
import sys, numpy as np, pandas as pd
g = sys.argv[1]
R = './raw/'; O = './work/'
meta = pd.read_csv(f'{R}{g}/{g}_meta.csv', index_col=0)
meta['Age'] = pd.to_numeric(meta['Age'], errors='coerce')
meta = meta.dropna(subset=['Age']).set_index('GSM_number')
hdr = open(f'{R}{g}/{g}_count_matrix.csv').readline().rstrip('\n').split('\t')
cols = [c for c in hdr]
keep = [c for c in cols if c in meta.index]
age = meta.loc[keep, 'Age'].astype(float).values
idx = [cols.index(c) for c in keep]
res = []
for ch in pd.read_csv(f'{R}{g}/{g}_count_matrix.csv', sep='\t', skiprows=1, header=None, index_col=0,
                      chunksize=20000, dtype={0: str}):
    X = ch.iloc[:, idx].to_numpy(np.float64)
    m = ~np.isnan(X); n = m.sum(1)
    Xa = np.where(m, X, 0.0); A = np.where(m, age[None, :], 0.0)
    sx = Xa.sum(1); sa = A.sum(1); sxx = (Xa ** 2).sum(1); saa = (A ** 2).sum(1); sxa = (Xa * A).sum(1)
    cov = sxa - sx * sa / n; vx = sxx - sx ** 2 / n; va = saa - sa ** 2 / n
    with np.errstate(invalid='ignore', divide='ignore'):
        r = cov / np.sqrt(vx * va)
        res.append(pd.DataFrame({'r': r, 'n': n, 'mean': sx / n, 'sd': np.sqrt(vx / (n - 1))}, index=ch.index))
S = pd.concat(res); S.index.name = 'cpg'
S.to_parquet(f'{O}{g}_stats.parquet')
meta.loc[keep, ['Gender', 'Age']].to_csv(f'{O}{g}_pheno.csv')
print(g, S.shape, len(keep), 'ages', age.min(), age.max())
