# Release checklist

- [x] Record Kisung You as the sole author in `CITATION.cff` and package
  metadata. No DOI is assigned.
- [x] License the BMTS software under the MIT License while keeping the bundled
  dataset's separately documented CC0 status explicit.
- [x] Execute all four notebooks and retain their embedded outputs and figures.
- [x] Run `python tests/test_bmts.py` and `python bmts.py`.
- [x] Audit the complete public tree for absolute user paths, credential
  signatures, raw preparation caches, hidden build artifacts, and private data.
- [ ] Create a GitHub repository from `public/`, not from the full research
  workspace.
- [ ] Tag the first public release if versioned archival is desired.
