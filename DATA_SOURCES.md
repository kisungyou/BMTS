# Data sources and redistribution

The two simulation notebooks generate synthetic observations from fixed seeds.
The iris notebook uses Fisher's iris data as distributed with scikit-learn.

The hematopoietic experiment uses the Nestorowa et al. mouse hematopoietic stem
and progenitor cell dataset, GEO accession `GSE81682` (PubMed `27365425`). The
included derivative comes from the Bioconductor `scRNAseq` asset
`nestorowa-hsc-2016`, pinned to the public 2024-04-18 asset version. Its source
registry records the dataset license as CC0.

`data/nestorowa_bmts_input.npz` is a compact, pickle-free derivative containing:

- nine standardized RNA principal components for 1,656 QC-retained cells;
- ten index-sorting FACS measurements, with missing values retained as `NaN`;
- author phenotype annotations and sort gates; and
- library size, detected-gene, and ERCC quality-control quantities.

The same cells have RNA and FACS measurements. FACS values and phenotype labels
are withheld from mixture fitting and enter only the cross-modality evaluation.
Among the 1,656 retained cells, 1,430 have complete FACS measurements and 226
are excluded from that evaluation rather than imputed.

The derivative has SHA-256 digest
`bba96dd4a725990fab0a54cde49571bf97316466d85e6595dd47864442fee302`.
Complete source links, official QC reproduction, preprocessing settings,
software versions, array schemas, and member-level checksums are preserved in
`data/nestorowa_source.json`.

The Nestorowa source registry records the dataset as CC0. The BMTS software is
separately licensed under the MIT License; see `LICENSE`.

Primary references:

- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81682
- PubMed: https://pubmed.ncbi.nlm.nih.gov/27365425/
- Bioconductor scRNAseq: https://bioconductor.org/packages/release/data/experiment/html/scRNAseq.html
- OSCA Nestorowa workflow: https://bioconductor.org/books/3.21/OSCA.workflows/nestorowa-mouse-hsc-smart-seq2.html
