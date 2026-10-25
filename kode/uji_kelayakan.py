import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from data import baca_bgr, baca_json, jalur_relatif, peran_citra, rerata_cara_ta, semua_citra, tulis_json
from model import (JUMLAH_PERMUTASI, TAMBAHAN_UJUNG, TITIK_SUMBU, akurasi_1nn_kemiripan, akurasi_5lipat_berulang,
                   akurasi_per_kelompok, ambil_pita, daftar_klasifier, fitur_latar, fitur_rerata_ta, fraksi_jenuh,
                   histogram_hsv, interseksi_ta, p_permutasi_kemiripan, p_permutasi_loo, prediksi_loo,
                   profil_ternormalisasi, selang_wilson)

KELUARAN = "uji_kelayakan.json"
LEBAR_RENTANG_KASAR_NM = 300.0


def kumpulkan_data():
    baris = []
    for jalur in semua_citra():
        peran = peran_citra(jalur)
        if peran["peran"] != "foto spektrum":
            continue
        bgr = baca_bgr(jalur)
        pita, info = ambil_pita(bgr)
        rerata = rerata_cara_ta(bgr)
        pita_bgr = np.clip(pita[:, :, ::-1], 0, 255).astype(np.uint8)
        baris.append({"jalur": jalur_relatif(jalur), "kelas": peran["kelas"], "lensa": peran["lensa"], **info,
                      "histogram": {"seluruh citra": histogram_hsv(bgr), "dalam pita": histogram_hsv(pita_bgr)},
                      "fraksi_jenuh_pita": fraksi_jenuh(pita),
                      "fitur": {"profil pita (3 x 64)": profil_ternormalisasi(pita), "rerata RGB-HSV dalam pita": fitur_rerata_ta(pita),
                                "rerata RGB-HSV latar (kontrol negatif)": fitur_latar(bgr),
                                "rerata RGB-HSV seluruh citra (cara TA)": np.array(list(rerata.values()))}})
    return baris


def nilai_loo(klasifier, X, y, kelas, dengan_permutasi=True):
    tebakan = prediksi_loo(klasifier, X, y)
    benar = int(np.sum(tebakan == y))
    akurasi = benar / len(y)
    lipat = akurasi_5lipat_berulang(klasifier, X, y)
    return {"loo_benar": benar, "n": len(y), "loo_akurasi": akurasi, "loo_wilson95": selang_wilson(benar, len(y)),
            "loo_akurasi_seimbang": float(balanced_accuracy_score(y, tebakan)),
            "loo_matriks_kebingungan": confusion_matrix(y, tebakan, labels=kelas).tolist(),
            "p_permutasi_loo": p_permutasi_loo(klasifier, X, y, akurasi) if dengan_permutasi else None,
            "lipat5_rerata": float(np.mean(lipat)), "lipat5_min": float(np.min(lipat)), "lipat5_maks": float(np.max(lipat))}


def alih_kondisi(klasifier, X, y, lensa):
    latih, uji = lensa == "f15", lensa == "f20"
    tebakan = klasifier.fit(X[latih], y[latih]).predict(X[uji])
    return {"latih": "f15", "uji": "f20", "n_uji": int(uji.sum()), "benar": int(np.sum(tebakan == y[uji])),
            "pasangan": [[a, b] for a, b in zip(y[uji].tolist(), tebakan.tolist())]}


def kelompok_hampir_identik(baris):
    induk = {b["jalur"]: b["jalur"] for b in baris}

    def akar(j):
        while induk[j] != j:
            j = induk[j]
        return j

    for p in baca_json("uji_inventaris.json")["pasangan_hampir_sama"]:
        if p["a"] in induk and p["b"] in induk:
            induk[akar(p["a"])] = akar(p["b"])
    return np.array([akar(b["jalur"]) for b in baris])


def baseline_interseksi(baris, y):
    hasil = {}
    for wilayah in baris[0]["histogram"]:
        hist = [b["histogram"][wilayah] for b in baris]
        kemiripan = np.array([[interseksi_ta(a, c) for c in hist] for a in hist])
        akurasi = akurasi_1nn_kemiripan(kemiripan, y)
        benar = round(akurasi * len(y))
        hasil[f"histogram interseksi HSV {wilayah} (cara TA) | 1-NN"] = {
            "loo_benar": benar, "n": len(y), "loo_akurasi": akurasi, "loo_wilson95": selang_wilson(benar, len(y)),
            "p_permutasi_loo": p_permutasi_kemiripan(kemiripan, y, akurasi)}
    return hasil


def per_adegan(baris, y, kelompok):
    hasil = {}
    for nama_fitur in ("profil pita (3 x 64)", "rerata RGB-HSV dalam pita"):
        X = np.array([b["fitur"][nama_fitur] for b in baris])
        for nama_klasifier, klasifier in daftar_klasifier().items():
            hasil[f"{nama_fitur} | {nama_klasifier}"] = {"benar": akurasi_per_kelompok(klasifier, X, y, kelompok), "n": len(y)}
    return hasil


def main():
    baris = kumpulkan_data()
    y = np.array([b["kelas"] for b in baris])
    lensa = np.array([b["lensa"] for b in baris])
    kelas = sorted(set(y))
    hasil_kelas, hasil_lensa, hasil_alih = {}, {}, {}
    for nama_fitur in baris[0]["fitur"]:
        X = np.array([b["fitur"][nama_fitur] for b in baris])
        for nama_klasifier, klasifier in daftar_klasifier().items():
            kunci = f"{nama_fitur} | {nama_klasifier}"
            hasil_kelas[kunci] = nilai_loo(klasifier, X, y, kelas)
            hasil_lensa[kunci] = nilai_loo(klasifier, X, lensa, ["f15", "f20"], dengan_permutasi=False)
            hasil_alih[kunci] = alih_kondisi(klasifier, X, y, lensa)
    kelompok = kelompok_hampir_identik(baris)
    panjang_warna = {l: [b["panjang_px"] / (1 + 2 * TAMBAHAN_UJUNG) for b in baris if b["lensa"] == l] for l in ("f15", "f20")}
    tulis_json(KELUARAN, {"kelas": kelas, "n_per_kelas": {k: int(np.sum(y == k)) for k in kelas},
                          "n_per_kelas_lensa": {k: {l: int(np.sum((y == k) & (lensa == l))) for l in ("f15", "f20")} for k in kelas},
                          "jumlah_permutasi": JUMLAH_PERMUTASI, "titik_sumbu": TITIK_SUMBU, "peluang_acak_kelas": 1 / len(kelas),
                          "kelas_mayoritas": max(np.mean(y == k) for k in kelas),
                          "prediksi_kelas": hasil_kelas, "baseline_interseksi": baseline_interseksi(baris, y),
                          "jumlah_adegan": len(set(kelompok)), "tinggalkan_satu_adegan": per_adegan(baris, y, kelompok), "prediksi_lensa": hasil_lensa, "alih_f15_ke_f20": hasil_alih,
                          "panjang_pita_berwarna_px_median": {l: float(np.median(v)) for l, v in panjang_warna.items()},
                          "nm_per_px_kasar_asumsi_400_700": {l: LEBAR_RENTANG_KASAR_NM / float(np.median(v)) for l, v in panjang_warna.items()},
                          "fraksi_jenuh_pita_maks": max(b["fraksi_jenuh_pita"] for b in baris),
                          "citra": [{k: v for k, v in b.items() if k not in ("fitur", "histogram")} for b in baris]})


if __name__ == "__main__":
    main()
