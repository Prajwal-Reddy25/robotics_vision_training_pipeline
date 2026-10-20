# Reproducibility and published evidence

Updated 2026-10-20. Released **v1.2.0** is commit
`856e0574098a38dfaa1374cde1fcc3e0c923397b`; frozen experiments were measured on
2026-10-05/06. The original [evidence](evidence/v1_2/SHA256.json), datasets,
checkpoint hashes and findings are unchanged by this maintenance review.

## Offline recorded-evidence review

The maintenance checkout adds a standalone standard-library verifier. It is not
in the original v1.2.0 tag. Python 3.10+ suffices; no pip installation, GPU, network,
original dataset, model checkpoint or pickle deserialization is needed:

```bash
python -I -S scripts/verify_published_v1_2.py
python -I -S scripts/check_docs.py
```

`-I -S` isolates the interpreter from environment paths and site packages. The
verifier defaults to its checkout's root, independent of the working directory.
A reviewer can also point this maintenance script at an extracted published source
archive without changing that archive:

```bash
python -I -S /path/to/maintenance/scripts/verify_published_v1_2.py --repo-root /path/to/v1.2.0-source
```

It reads the frozen v1.1/v1.2 evidence only and checks:

- SHA256/size and inventory coverage for 63 files; original pretty-printed JSON
  serialization hashes are reconstructed from the lossless compact JSON.
- Complete 42-model / 84-size-evaluation identities, all 6,720 per-image records,
  annotation-derived class supports, ignored/unsupported class handling, native
  confusion sums, IoU/precision/recall/accuracy and comparable ID−real gaps.
- Projected synthetic matrices, 28 three-seed summaries, 24 paired contrasts,
  48 descriptive correlations, all per-run CSV values and raw timing quantiles.
- Gallery score/credit consistency, retaining the two unresolved creators.

The arithmetic uses explicit confusion-matrix sums in independent code, not the
production PyTorch/NumPy metric implementation. Float comparison tolerance is
1e-12 absolute/relative. Counts must be nonnegative integers. A failed check exits
nonzero, including under `python -O`. Reads are capped at 8 MiB per file, 16 MiB
and 100 inventoried files per evidence version. The CI job has a five-minute bound;
it installs no ML packages and never invokes the training/download scripts.

**What this establishes:** consistency of the published numbers, artifact change
detection and preservation of all recorded seeds/arms. It does not authenticate
measurements, establish original image/checkpoint byte identity from unavailable
files, rerun inference, independently measure latency, or check live rights/links.
An inventory can be rewritten together with its files; use the published tag/Git
history as a reference. Corruption tests therefore also rewrite hashes deliberately
and check that incorrect arithmetic and omitted runs still fail.

Documentation validation checks local Markdown/HTML file links and simple heading
anchors. It skips code-fence examples and external URLs; it is not an HTML renderer
or a continuous external-link availability monitor.

## Full experiment reproduction

The original [real-data reproduction procedure](REAL_DATA_GENERALIZATION_V1_2.md#reproduction)
regenerates the synthetic study, downloads the licensed COCO subset and evaluates
all models. Model binaries and original photos are intentionally Git-ignored;
the GitHub Release has no downloadable trained-checkpoint assets. Frozen original
checkpoint hashes cannot be independently verified from the source archive alone.
The original `scripts/review_real_v1_2.py` checks actual local images/checkpoints
and is a separate, stronger check when those files exist.

Use the original source/configuration and captured environment before comparing
weights/metrics. No retraining or new experiment is required for the offline review.
Changed upstream photos/labels must fail the existing frozen-reference identity gate.
Serialization/path metadata may change file-byte hashes across regenerated training
outputs; distinguish weight/count reproduction from arbitrary renamed file identity.

## Dependencies and packaging

Package metadata and `robotics_vision.__version__` remain **1.2.0** on this branch.
New scripts/CI/docs are maintenance tooling, not additions to the wheel's training
or inference API. `MANIFEST.in` now includes configurations, scripts, constraints,
documentation and the bounded published evidence in the Python source distribution;
raw data, generated runs and checkpoint/deployment binaries are excluded. The
runtime wheel remains small. This fixes source distributions containing tests
without their required repository context. A v1.2.1 patch is proposed, not created automatically.

The original `pyproject.toml` intentionally has broad compatible dependency ranges;
CI installs current matching versions. This is a compatibility test, not an exact
environment lock. Frozen [CPU](evidence/v1_2/cpu_versions.txt) and
[CUDA](evidence/v1_2/cuda_versions.txt) captures include a machine-specific editable
path and must not be installed verbatim on another machine.

[Portable CPU constraints](../constraints/v1_2_cpu.txt) retain the captured package
pins and exclude the editable path. For the recorded Linux / Python 3.12.3 CPU stack:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints/v1_2_cpu.txt -e '.[all]'
python -m pip check
```

Constraints restrict versions; they do not install every captured transitive package,
provide wheel hashes, pin Python/OS/toolchain/hardware, or guarantee wheel availability
on every platform. Build-system requirements and CI action versions remain broadly
pinned. This review does not claim cross-platform, cross-version or general CUDA
bitwise determinism. The retained v1.1 CUDA resume discrepancy remains relevant.

## Actual validation versus history

The [dated maintenance review](MAINTENANCE_REVIEW_2026_10_20.md) records new local
checks, the observed published-tag CI run, and unperformed checks separately.
[Original release validation](RELEASE_READINESS_V1_2.md) preserves the October 6
experiment record. No historical CPU/GPU timing, accuracy or test count is recast
as a new measurement.
