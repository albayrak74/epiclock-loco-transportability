"""LOCO EN/GBM/ensemble + single-cohort EN; saves individual predictions."""
from common import *
rows = []
for h in C:
    tr = [g for g in C if g != h]
    en = fit_en(tr); gb = fit_gbm(tr); gbu = fit_gbm(tr, transform=False)
    d = PH[h].copy(); d['cohort'] = h
    d['EN'] = pred_en(en, h); d['GBM'] = pred_gbm(gb, h); d['GBM_raw'] = pred_gbm(gbu, h)
    d['ENS'] = (d.EN + d.GBM) / 2
    d['EN_alpha'] = en[0].alpha_; d['EN_nonzero'] = int((en[0].coef_ != 0).sum())
    rows.append(d)
    print(h, 'EN MAE %.2f GBM %.2f ENS %.2f GBMraw %.2f nonzero %d' % (
        (d.EN - d.Age).abs().mean(), (d.GBM - d.Age).abs().mean(), (d.ENS - d.Age).abs().mean(),
        (d.GBM_raw - d.Age).abs().mean(), d.EN_nonzero.iloc[0]), flush=True)
P = pd.concat(rows)
sc = fit_en(['GSE40279'])
P['SC'] = np.nan
for h in C:
    if h != 'GSE40279':
        P.loc[P.cohort == h, 'SC'] = pred_en(sc, h)
        print(h, 'single-cohort MAE %.2f' % (P[P.cohort == h].SC - P[P.cohort == h].Age).abs().mean())
P.to_csv(W + 'loco_predictions.csv')
