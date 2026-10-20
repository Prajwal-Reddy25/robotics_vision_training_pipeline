# v1.2.1 — Evidence Verification and Packaging

This maintenance release improves independent verification,
packaging completeness, and documentation of the v1.2 results.

## Improvements

- Offline verification of recorded v1.2 evidence.
- Corruption-detection regression tests.
- Documentation-link validation.
- Source-distribution completeness through MANIFEST.in.
- Portable CPU dependency constraints.
- Improved README, results gallery, and release documentation.

## Validation

The maintenance review reported 91 passing CPU tests,
one CUDA-only skip, successful Ruff and packaging checks,
and 141 validated local documentation links.

The release CI includes independent evidence verification
and the existing CPU test-and-smoke workflow.

## Scope

No models were retrained.
Frozen experimental results and the v1.2.0 tag are unchanged.
CUDA was not rerun for this maintenance patch.

See docs/MAINTENANCE_REVIEW_2026_10_20.md.
