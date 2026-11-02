# Research journal

Write the **hypothesis before** each experiment and the **result after**. This is the project's memory: the report's Ablations and Limits sections come straight from here.

---

## Decisions already made

| Date | Decision | Why |
|---|---|---|
| 2026-10-26 | Split train / val / test by `lesion_id` (70 / 10 / 20, stratified by class) | One lesion can have several photos. A random split puts the same lesion in train and test and inflates the scores. |
| 2026-10-26 | Work at 28×28 first, 64×64 only for the best model | Fast experiments on a free T4. |
| 2026-10-26 | Build our own arrays from the raw images, not from the `hmnist_*.csv` files | The hmnist files have no `lesion_id`, so a clean split is impossible. |
| 2026-10-26 | Class order `akiec, bcc, bkl, df, mel, nv, vasc` | Alphabetical, fixed once in `config.py`. |
| 2026-10-26 | Evaluate classification with balanced accuracy, macro F1, per-class recall | Answering "nv" every time already gives about 67 % accuracy. |

---

## Compute log

Fill one row per run. GPU hours are reported in the appendix of the report.

| Date | Run name | Script + arguments | Platform | GPU time | Notes |
|---|---|---|---|---|---|
| | | | | | |

---

## Experiments

Copy this block for each experiment.

### YYYY-MM-DD · Short title

- **Hypothesis (written before):**
- **Setup:** script, arguments, seed(s)
- **Expected result:**
- **Actual result:** numbers + path to the figure
- **Conclusion:** confirmed / refuted, and what it changes next

---

## Problems and fixes

| Date | Problem | Fix |
|---|---|---|
| | | |
