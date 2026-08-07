# Public data bundle

- `nestorowa_bmts_input.npz` is the 399 KB, pickle-free derivative used by the
  hematopoietic notebook. `bmts.load_nestorowa()` validates its principal shape
  and always loads it with `allow_pickle=False`.
- `nestorowa_source.json` is the immutable provenance, preprocessing, schema,
  and checksum registry for that derivative.
- `paper_scale_summary.json` contains the numerical summaries and settings from
  the manuscript-scale runs for Simulation 1, Simulation 2, iris, and the HSPC
  experiment, together with deterministic validation checks.

See `../DATA_SOURCES.md` before redistributing the data or publishing a derived
repository.

