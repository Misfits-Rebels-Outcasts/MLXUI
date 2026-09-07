# Sample assets — provenance (CFM-R11-0b)

Every file in this directory ships inside the app and is used to seed a
newly-added `Read *` row so it runs without further input. All assets are
owned by the project (generated in the owner's `catflow-mlx` fixture
pipeline or written fresh for this feature). None are third-party or
copyrighted works.

| File | Provenance |
|---|---|
| `sample-audio.m4a` | Copied from `catflow-mlx/gallery_fixtures/01-SpokenSummary/memo.m4a` — a generated ~6 s speech clip (same audio `01-SpokenSummary` ships). |
| `sample-text.txt` | Written fresh for this feature. |
| `sample-image.png` | Copied from `catflow-mlx/gallery_fixtures/21-PhotoWebPrep/vacation/photo-01.png` — a generated photo. |
| `sample-images/sample-img-01/02/03.png` | Copied from `21-PhotoWebPrep/vacation/photo-01/02/03.png` — generated photos. |
| `sample-files/sample-note-01/02/03.txt` | Written fresh for this feature. |
| `sample-pdf.pdf` | Generated from text via `cupsfilter` — a one-page document with real selectable text (no fonts, no scans). |
| `sample-data.csv` | Written fresh for this feature. |
| `sample-data.json` | Written fresh for this feature. |
| `PersonalBudget.png` | Owner-supplied screenshot of a personal budget table (1308×646), used as the "Use Sample Image" default in the OCR Run UI (`OCRRunView`). Not a Read-row seed. |
