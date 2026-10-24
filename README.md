# epiclock-loco-transportability

Analysis code for:

> Alkan İ, Albayrak M. *Age-range coverage and cohort diversity limit the transportability of machine-learning epigenetic clocks: a leave-one-cohort-out benchmark of 1,182 blood methylomes.* (submitted)

## Data
The five whole-blood Illumina 450K cohorts (GSE36054, GSE40279, GSE41169, GSE64495, GSE69270) are taken as distributed by
MethAgingDB (Zenodo, doi: [10.5281/zenodo.15714493](https://doi.org/10.5281/zenodo.15714493)). Download the five
`GSExxxxx.tar.gz` archives and extract them into `raw/`.

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1 | `s1_screen.py GSExxxxx` (run for each cohort) | per-CpG correlation with age, n, mean, SD |
| 2 | `s2_select.py` | leakage-free CpG selection for all 31 training sets; panel matrices |
| 3 | `s3_loco.py` | leave-one-cohort-out elastic net, gradient boosting, ensemble and single-cohort predictions |
| 4 | `s4_analyses.py` | bootstrap CIs, calibration slopes, age-band errors |
| 5 | `s5_more.py [div few sex cell pub final]` | training-cohort diversity (75 models), few-shot recalibration, sex residuals, cell composition, published clocks, final clock |
| 6 | `f_figs.py` | Figures 1–4 |

Run the scripts from the repository root; the paths `raw/`, `work/` and `fig/` are relative and set at the top of each script.

## Software
Python 3.13, scikit-learn 1.9.1, numpy, pandas, scipy, statsmodels, pyarrow, matplotlib, biolearn (published clock
coefficients and the Reinius reference). The elastic net is fitted with `ElasticNetCV(l1_ratio=0.5, alphas=30, cv=KFold(5),
max_iter=200000, precompute=True)`; this setting converges without warnings. Random seeds are fixed (2026 for bootstrap and
few-shot draws, 0 for gradient boosting).

## License
MIT
