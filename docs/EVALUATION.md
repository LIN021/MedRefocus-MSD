# Evaluation

```bash
medrefocus evaluate --manifest data/fold0.csv --predictions runs/fold0/final \
  --output runs/fold0/evaluation --bootstrap 10000
```

The default evaluates `role=test`. Use `--role all` for a test-only manifest without roles. Reference labels are accessed only by training and evaluation entry points. Native NIfTI size, spacing, origin and direction must agree; mismatches fail rather than being silently resampled.

| Quantity | Definition |
|---|---|
| Tumor Dice | Whole-volume overlap for label 2 |
| NSD | Area-weighted normalized surface Dice at 2 mm |
| HD95 | Area-weighted robust Hausdorff distance, mm |
| Complete miss | Nonempty reference tumor with zero prediction–reference intersection, including FP-only predictions |
| Recovery | Baseline complete miss with positive final overlap |
| New miss | Baseline positive overlap lost by the final method |
| Recovery quality | Initial misses whose final Dice reaches ≥0.25 or ≥0.50 |
| Severe decline | Final minus baseline Dice or NSD strictly below −0.05 (−5 pp) |

Both-empty overlap/surface scores are undefined (`NaN`). A single empty mask gives Dice/NSD 0 and HD95 infinity. JSON summary HD95 means use the same finite patient subset for the baseline and final predictions, and include its count. Comparisons across more than two methods require intersecting finite patients **across all methods**, not reusing different pairwise subsets.

`patients.csv` stores per-patient values internally on a 0–1 scale for Dice/NSD. `summary.json` reports their means in percent and paired gains/95% patient-bootstrap intervals in percentage points. Patient-level mean scores are used; the metric does not pool all voxels across patients. `--bootstrap 0` skips interval computation.

The lower-level module `python -m medrefocus.evaluation.evaluate_segmentation --help` additionally supports generic per-label and lesion analysis. The MSD entry point above defines the paired recovery comparison used in this repository.
