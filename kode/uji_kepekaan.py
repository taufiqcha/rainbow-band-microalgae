import numpy as np
from joblib import Parallel, delayed

from data import KANAL, baca_bgr, foto_berlabel, jalur_relatif, tulis_json
from model import (BATAS_KROMA, FRAKSI_KROMA, PERSENTIL_INTI, TITIK_LINTANG, TITIK_SUMBU, ambil_pita, daftar_klasifier,
                   geometri_pita, prediksi_loo, profil_ternormalisasi)

KELUARAN = "uji_kepekaan.json"
SAPUAN = {"BATAS_KROMA": ("batas_kroma", BATAS_KROMA, (10, 15, 20, 25, 30, 35, 40, 50, 60)),
          "FRAKSI_KROMA": ("fraksi_kroma", FRAKSI_KROMA, (0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.7)),
          "PERSENTIL_INTI": ("persentil_inti", PERSENTIL_INTI, (30, 40, 50, 60, 70, 80, 90)),
          "TITIK_SUMBU": ("titik", TITIK_SUMBU, (16, 32, 48, 64, 96, 128)),
          "TITIK_LINTANG": ("titik_lintang", TITIK_LINTANG, (4, 8, 12, 16, 24, 32))}
KUNCI_GRID = ("titik", "titik_lintang")
KANAL_PROFIL = KANAL[:3]


def daftar_setelan():
    return [(nama, nilai, {kunci: nilai}) for nama, (kunci, _, deret) in SAPUAN.items() for nilai in deret]


def pelokalan_satu(bgr, setelan):
    _, _, geometri = geometri_pita(bgr, **{k: v for k, v in setelan.items() if k not in KUNCI_GRID})
    pita, _ = ambil_pita(bgr, geometri=geometri, **{k: v for k, v in setelan.items() if k in KUNCI_GRID})
    return {"profil": profil_ternormalisasi(pita)}


def ukur_foto(jalur):
    bgr = baca_bgr(jalur)
    return pelokalan_satu(bgr, {}), [pelokalan_satu(bgr, setelan) for _, _, setelan in daftar_setelan()]


def ke_titik_bawaan(profil):
    kanal = profil.reshape(len(KANAL_PROFIL), -1)
    posisi = np.linspace(0, 1, kanal.shape[1])
    return np.array([np.interp(np.linspace(0, 1, TITIK_SUMBU), posisi, k) for k in kanal])


def korelasi_profil(hasil, bawaan):
    pasangan = [(ke_titik_bawaan(h["profil"]), b["profil"].reshape(len(KANAL_PROFIL), -1)) for h, b in zip(hasil, bawaan)]
    gabungan = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in pasangan])
    per_kanal = np.array([[np.corrcoef(a[i], b[i])[0, 1] for i in range(len(KANAL_PROFIL))] for a, b in pasangan])
    return gabungan, per_kanal


def loo_benar(X, y):
    return int(np.sum(prediksi_loo(daftar_klasifier()["1-NN"], X, y) == y))


def nilai_setelan(hasil, bawaan, jalur, y, f15):
    X = np.array([h["profil"] for h in hasil])
    korelasi, per_kanal = korelasi_profil(hasil, bawaan)
    return {"loo_1nn_benar": loo_benar(X, y),
            "f15_loo_1nn_benar": loo_benar(X[f15], y[f15]), "korelasi_profil_min": float(korelasi.min()),
            "foto_korelasi_min": jalur[int(np.argmin(korelasi))],
            "korelasi_per_kanal_min": {k: float(per_kanal[:, i].min()) for i, k in enumerate(KANAL_PROFIL)},
            "foto_korelasi_per_kanal_min": {k: jalur[int(np.argmin(per_kanal[:, i]))] for i, k in enumerate(KANAL_PROFIL)}}


def main():
    foto = foto_berlabel()
    jalur = [jalur_relatif(j) for j, _ in foto]
    y = np.array([p["kelas"] for _, p in foto])
    f15 = np.array([p["lensa"] == "f15" for _, p in foto])
    ukur = Parallel(n_jobs=-1)(delayed(ukur_foto)(j) for j, _ in foto)
    bawaan = [u[0] for u in ukur]
    X_bawaan = np.array([b["profil"] for b in bawaan])
    sapuan = {nama: {"bawaan": bawaan_nilai, "hasil": []} for nama, (_, bawaan_nilai, _) in SAPUAN.items()}
    for i, (nama, nilai, _) in enumerate(daftar_setelan()):
        sapuan[nama]["hasil"].append({"nilai": nilai, **nilai_setelan([u[1][i] for u in ukur], bawaan, jalur, y, f15)})
    for isi in sapuan.values():
        isi["loo_1nn_benar_min"] = min(h["loo_1nn_benar"] for h in isi["hasil"])
        isi["f15_loo_1nn_benar_min"] = min(h["f15_loo_1nn_benar"] for h in isi["hasil"])
        isi["korelasi_profil_min"] = min(h["korelasi_profil_min"] for h in isi["hasil"])
        isi["korelasi_per_kanal_min"] = {k: min(h["korelasi_per_kanal_min"][k] for h in isi["hasil"]) for k in KANAL_PROFIL}
    tulis_json(KELUARAN, {"n": len(y), "n_f15": int(f15.sum()), "klasifier": "1-NN, LOO",
                          "korelasi": ("Pearson profil 3 x 64 per foto terhadap profil bawaan, digabung tiga kanal dan per kanal R, G, B; "
                                       "profil dengan TITIK_SUMBU lain diinterpolasi linear ke 64 titik per kanal"),
                          "bawaan": {"loo_1nn_benar": loo_benar(X_bawaan, y),
                                     "f15_loo_1nn_benar": loo_benar(X_bawaan[f15], y[f15])},
                          "sapuan": sapuan})


if __name__ == "__main__":
    main()
