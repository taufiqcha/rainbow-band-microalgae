import numpy as np
from scipy.stats import binomtest, f_oneway
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests

from data import SEED, tulis_json
from model import (TITIK_SUMBU, akurasi_per_kelompok, daftar_klasifier, interseksi_ta, p_permutasi_loo, prediksi_loo,
                   selang_wilson)
from uji_kelayakan import kumpulkan_data, kelompok_hampir_identik

KELUARAN = "uji_pembeda.json"
ULANGAN_ACAK = 200
BATAS_Q = 0.05
ALFA = 0.05
FITUR_UTAMA = "profil pita (3 x 64)"
PEMBANDING = ("rerata RGB-HSV seluruh citra (cara TA)", "rerata RGB-HSV dalam pita")


def statistik_f(X, y):
    kelas = sorted(set(y))
    f, p = f_oneway(*[X[y == k] for k in kelas])
    return np.nan_to_num(f), np.nan_to_num(p, nan=1.0)


def ringkas_f(X, y):
    f, p = statistik_f(X, y)
    q = multipletests(p, method="fdr_bh")[1]
    hasil = {}
    for i, kanal in enumerate("RGB"):
        bagian = slice(i * TITIK_SUMBU, (i + 1) * TITIK_SUMBU)
        bermakna = np.nonzero(q[bagian] < BATAS_Q)[0]
        hasil[kanal] = {"F_maks": float(f[bagian].max()), "titik_F_maks": int(np.argmax(f[bagian])),
                        "posisi_relatif_F_maks": float(np.argmax(f[bagian]) / (TITIK_SUMBU - 1)),
                        "jumlah_titik_q_kurang_005": int(len(bermakna)),
                        "rentang_titik_bermakna": [int(bermakna.min()), int(bermakna.max())] if len(bermakna) else None}
    return hasil


def mcnemar_eksak(benar_a, benar_b):
    a_saja = int(np.sum(benar_a & ~benar_b))
    b_saja = int(np.sum(~benar_a & benar_b))
    p = binomtest(min(a_saja, b_saja), a_saja + b_saja, 0.5).pvalue if a_saja + b_saja else 1.0
    return {"hanya_profil_benar": a_saja, "hanya_pembanding_benar": b_saja, "p_mcnemar_eksak": float(p)}


def prediksi_kemiripan(hist, y):
    kemiripan = np.array([[interseksi_ta(a, c) for c in hist] for a in hist])
    np.fill_diagonal(kemiripan, -np.inf)
    return y[np.argmax(kemiripan, axis=1)]


def u2_mcnemar(baris, y):
    klasifier = daftar_klasifier()["1-NN"]
    benar_profil = prediksi_loo(klasifier, np.array([b["fitur"][FITUR_UTAMA] for b in baris]), y) == y
    hasil = {}
    for nama in PEMBANDING:
        benar = prediksi_loo(klasifier, np.array([b["fitur"][nama] for b in baris]), y) == y
        hasil[f"{nama} | 1-NN"] = {"benar": int(benar.sum()), **mcnemar_eksak(benar_profil, benar)}
    for wilayah in ("seluruh citra", "dalam pita"):
        benar = prediksi_kemiripan([b["histogram"][wilayah] for b in baris], y) == y
        hasil[f"histogram interseksi HSV {wilayah} (cara TA) | 1-NN"] = {"benar": int(benar.sum()), **mcnemar_eksak(benar_profil, benar)}
    p_holm = multipletests([v["p_mcnemar_eksak"] for v in hasil.values()], method="holm")[1]
    for v, p in zip(hasil.values(), p_holm):
        v["p_holm"] = float(p)
    return {"profil_benar": int(benar_profil.sum()), "n": len(y), "koreksi": f"Holm atas {len(hasil)} pembanding", "alfa": ALFA, "pembanding": hasil}


def u5_dalam_f15(baris, y, lensa, kelompok):
    pilih = lensa == "f15"
    yf = y[pilih]
    hasil = {}
    for nama in (FITUR_UTAMA,) + PEMBANDING:
        X = np.array([b["fitur"][nama] for b in baris])[pilih]
        for nama_klasifier, klasifier in daftar_klasifier().items():
            benar = int(np.sum(prediksi_loo(klasifier, X, yf) == yf))
            hasil[f"{nama} | {nama_klasifier}"] = {
                "loo_benar": benar, "n": int(len(yf)), "loo_wilson95": selang_wilson(benar, len(yf)),
                "p_permutasi_loo": p_permutasi_loo(clone(klasifier), X, yf, benar / len(yf)),
                "tinggalkan_satu_kelompok_hampir_identik_benar": akurasi_per_kelompok(klasifier, X, yf, kelompok[pilih]),
                "jumlah_kelompok_hampir_identik": int(len(set(kelompok[pilih])))}
    return {"n_per_kelas": {k: int(np.sum(yf == k)) for k in sorted(set(yf))}, "hasil": hasil}


def acak_per_foto(X, benih):
    acak = np.random.default_rng(benih)
    kubus = X.reshape(len(X), 3, TITIK_SUMBU).copy()
    for i in range(len(X)):
        kubus[i] = kubus[i][:, acak.permutation(TITIK_SUMBU)]
    return kubus.reshape(len(X), -1)


def u6_urutan_acak(baris, y):
    X = np.array([b["fitur"][FITUR_UTAMA] for b in baris])
    hasil = {}
    for nama_klasifier, klasifier in daftar_klasifier().items():
        asli = int(np.sum(prediksi_loo(klasifier, X, y) == y))
        acak = [int(np.sum(prediksi_loo(klasifier, acak_per_foto(X, SEED + i), y) == y)) for i in range(ULANGAN_ACAK)]
        urut = np.sort(X.reshape(len(X), 3, TITIK_SUMBU), axis=2).reshape(len(X), -1)
        rerata_kanal = X.reshape(len(X), 3, TITIK_SUMBU).mean(axis=2)
        hasil[nama_klasifier] = {"asli_benar": asli, "acak_rerata_benar": float(np.mean(acak)), "acak_min": int(min(acak)),
                                 "acak_maks": int(max(acak)),
                                 "nilai_diurutkan_benar": int(np.sum(prediksi_loo(klasifier, urut, y) == y)),
                                 "rerata_kanal_saja_benar": int(np.sum(prediksi_loo(klasifier, rerata_kanal, y) == y))}
    return {"n": len(y), "ulangan_acak": ULANGAN_ACAK, "hasil": hasil}



def main():
    baris = kumpulkan_data()
    y = np.array([b["kelas"] for b in baris])
    lensa = np.array([b["lensa"] for b in baris])
    kelompok = kelompok_hampir_identik(baris)
    X = np.array([b["fitur"][FITUR_UTAMA] for b in baris])
    f15 = lensa == "f15"
    ringkas_semua = ringkas_f(X, y)
    ringkas_f15 = ringkas_f(X[f15], y[f15])
    ringkas_lensa = ringkas_f(X, lensa)
    profil_kelas = {f"{k} | {l}": {"n": int(np.sum((y == k) & (lensa == l))),
                                   "rerata": X[(y == k) & (lensa == l)].mean(axis=0).tolist(),
                                   "sd": X[(y == k) & (lensa == l)].std(axis=0, ddof=1).tolist() if np.sum((y == k) & (lensa == l)) > 1 else None}
                    for k in sorted(set(y)) for l in ("f15", "f20") if np.any((y == k) & (lensa == l))}
    tulis_json(KELUARAN, {
        "titik_sumbu": TITIK_SUMBU,
        "u1": {"batas_q_bh": BATAS_Q, "F_kelas_semua": ringkas_semua, "F_kelas_f15_saja": ringkas_f15,
               "F_lensa_semua": ringkas_lensa, "profil_kelas": profil_kelas},
        "u2": u2_mcnemar(baris, y),
        "u5": u5_dalam_f15(baris, y, lensa, kelompok),
        "u6": u6_urutan_acak(baris, y)})


if __name__ == "__main__":
    main()
