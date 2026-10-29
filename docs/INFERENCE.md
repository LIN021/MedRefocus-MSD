# Inference

Inference needs CT and native-grid Global/Auxiliary observer outputs. See [DATA.md](DATA.md) for the manifest. References can be omitted; if present in a training manifest they are explicitly removed from each prediction record before loading.

```bash
medrefocus predict --manifest data/inference.csv --models runs/fold0 \
  --output runs/inference --device cuda
```

Use `--role test` when the CSV contains fit, selection and test rows. Use `--device cpu` for a CPU run. Only load checkpoints from trusted sources: PyTorch/joblib model deserialization can execute Python objects.

Each model directory contains the fitted `ucrp.pt`, `expert.pt`, `pcsc.joblib` and `config.json`. The ranker and forest carry ordered feature schemas; incompatible schemas are rejected.

The output contains native-grid `<case>.nii.gz` label maps and `inference_records.json`. Records include the half-open ROI in `[z0,z1,y0,y1,x0,x1]` order, expert-call count, proposal threshold, PCSC score, acceptance flag and actual final added-voxel count.

The Local Expert sees four channels in order: volume-normalized CT, Global pancreas probability, Global tumor probability and Global hard label divided by two. The selected native-spacing patch is processed once. Existing tumor voxels remain tumor in the proposal; outside the ROI the full Global map is unchanged. Rejecting a proposal returns an exact copy of the entire Global label map.

Candidate construction uses no CT intensities or reference labels. Its per-patient uniform sampler is seeded by SHA-256 of `seed:case`, making candidate choices stable under patient reordering. Ideal-gain and recovery supervision are used only during fitting.
