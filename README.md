# Rainbow-band localization

Code for *Low-Cost Diffraction Grating Camera for Microalgae Culture Classification Based on Rainbow-Band Localization*.

Photographs of Chlorella, Nannochloropsis, and Spirulina cultures taken through a transmission grating with a phone camera are classified by the color profile along the rainbow band. The band is located in every photograph by a chroma mask and a least median of squares line, sampled on a 64 x 16 grid, and compared with the image-wide mean color by leave-one-out 1-NN, exact McNemar tests, permutation tests, and robustness and sensitivity analyses. `results/` holds the computed results behind every number in the paper. Identifiers in the code are in Indonesian.

## Data

`data.zip` holds the 73 image files of the acquisition archive (Table S1). Extract it in the repository root to create `data/`. The 23 labeled spectrum photographs are in `data/Testing/F.15/` (15 cm lens, one folder per species) and `data/Testing/f.20/` (20 cm lens).

## Run

Python 3.11.

```bash
pip install -r requirements.txt
unzip data.zip
cd kode
python3 uji_inventaris.py
python3 uji_kelayakan.py
python3 uji_pembeda.py
python3 uji_kekokohan.py
python3 uji_kepekaan.py
python3 uji_pelokalan.py
python3 uji_garis_dasar.py
python3 gambar_pelokalan_pita.py
python3 gambar_profil_kelas.py
```

## Scripts

| Script | Result file | Paper |
|---|---|---|
| `uji_inventaris.py` | `uji_inventaris.json` | Tables S1 and S2, near-identical photograph groups |
| `uji_kelayakan.py` | `uji_kelayakan.json` | Table II, Tables S4 and S5 |
| `uji_pembeda.py` | `uji_pembeda.json` | McNemar tests, 15 cm lens only, point order, Table S6 |
| `uji_kekokohan.py` | `uji_kekokohan.json` | Table III, Table S7, lane profiles |
| `uji_kepekaan.py` | `uji_kepekaan.json` | Table S8 |
| `uji_pelokalan.py` | `uji_pelokalan.json`, `results/gambar/e2_3_tumpang_pita.pdf` | Band found per photograph, Table S9, Fig. S1 |
| `uji_garis_dasar.py` | `uji_garis_dasar.json` | Table II, majority-class rule |
| `gambar_pelokalan_pita.py` | `figures/Figure_1b.pdf` | Fig. 1(b) |
| `gambar_profil_kelas.py` | `figures/Figure_2.pdf` | Fig. 2 |

`model.py` holds the band localization, features, classifiers, and validation, and `data.py` the paths, labels, and file readers. Table S3 lists the constants of `model.py`. Fig. 1(a) is a diagram drawn in draw.io and is not generated here.
