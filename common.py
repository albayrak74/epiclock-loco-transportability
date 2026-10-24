import json, numpy as np, pandas as pd, warnings
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
W = './work/'
C = ['GSE36054', 'GSE40279', 'GSE41169', 'GSE64495', 'GSE69270']
SEL = json.load(open(W + 'selections.json'))
PANEL = {g: pd.read_parquet(f'{W}{g}_panel.parquet') for g in C}
PH = {g: pd.read_csv(f'{W}{g}_pheno.csv', index_col=0) for g in C}


def F(a, adult=20):
    a = np.asarray(a, float)
    return np.where(a <= adult, np.log(a + 1) - np.log(adult + 1), (a - adult) / (adult + 1))


def Finv(x, adult=20):
    x = np.asarray(x, float)
    return np.where(x < 0, (1 + adult) * np.exp(x) - 1, (1 + adult) * x + adult)


def key(tr):
    return '+'.join(sorted(tr))


def data(cohorts, cpgs):
    X = pd.concat([PANEL[g][cpgs] for g in cohorts])
    y = np.concatenate([PH[g].Age.values for g in cohorts])
    return X, y


def fit_en(tr, n_alphas=30, folds=5):
    cp = SEL[key(tr)]
    X, y = data(sorted(tr), cp)
    mu = X.mean()
    m = ElasticNetCV(l1_ratio=0.5, alphas=n_alphas, cv=KFold(folds), max_iter=200000, precompute=True, n_jobs=2)
    m.fit(X.fillna(mu).values, F(y))
    return m, cp, mu


def pred_en(model, g):
    m, cp, mu = model
    return Finv(m.predict(PANEL[g][cp].fillna(mu).values))


def fit_gbm(tr, transform=True):
    cp = SEL[key(tr)]
    X, y = data(sorted(tr), cp)
    mu = X.mean()
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                      l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(X.fillna(mu).values, F(y) if transform else y)
    return m, cp, mu, transform


def pred_gbm(model, g):
    m, cp, mu, t = model
    p = m.predict(PANEL[g][cp].fillna(mu).values)
    return Finv(p) if t else p
