"""Step 2: leakage-free CpG selection for every training set; extract the union panel."""
import itertools, json, numpy as np, pandas as pd
W = './work/'; R = './raw/'
C = ['GSE36054', 'GSE40279', 'GSE41169', 'GSE64495', 'GSE69270']
S = {g: pd.read_parquet(f'{W}{g}_stats.parquet') for g in C}
common = sorted(set.intersection(*[set(S[g].index) for g in C]))
print('common CpGs', len(common))
Z = pd.DataFrame({g: np.arctanh(S[g].loc[common, 'r'].clip(-0.999999, 0.999999)) * np.sqrt(S[g].loc[common, 'n'] - 3) for g in C})
sel = {}
for k in range(1, 6):
    for tr in itertools.combinations(C, k):
        z = Z[list(tr)].sum(1) / np.sqrt(k)
        sel['+'.join(tr)] = z.abs().nlargest(2000).index.tolist()
union = sorted(set().union(*sel.values()))
print('training sets', len(sel), 'union', len(union))
json.dump(sel, open(W + 'selections.json', 'w'))
open(W + 'panel.txt', 'w').write('\n'.join(union))
BL = '/usr/local/lib/python3.13/dist-packages/biolearn/data/'
extra = set()
for f in ['Horvath1', 'Horvath2', 'Hannum', 'Lin']:
    extra |= set(pd.read_csv(BL + f + '.csv').CpGmarker)
extra |= set(pd.read_csv(BL + '450K_reinius_12_reference.csv', index_col=0).index)
extra = sorted(extra - set(union))
open(W + 'extra.txt', 'w').write('\n'.join(extra))
union = union + extra
U = set(union)
for g in C:
    ph = pd.read_csv(f'{W}{g}_pheno.csv', index_col=0)
    hdr = open(f'{R}{g}/{g}_count_matrix.csv').readline().rstrip('\n').split('\t')
    idx = [hdr.index(s) for s in ph.index]
    parts = []
    for ch in pd.read_csv(f'{R}{g}/{g}_count_matrix.csv', sep='\t', skiprows=1, header=None, index_col=0, chunksize=20000):
        ch = ch[ch.index.isin(U)]
        if len(ch): parts.append(ch.iloc[:, idx].astype(np.float32))
    M = pd.concat(parts).reindex(union).T
    M.index = ph.index
    M.to_parquet(f'{W}{g}_panel.parquet')
    print(g, M.shape, 'missing frac', float(M.isna().mean().mean()))
