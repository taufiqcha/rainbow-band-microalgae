import cv2
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from data import baca_bgr, jalur_relatif, peran_citra, semua_citra, tulis_json
from model import TITIK_LINTANG, TITIK_SUMBU, ambil_pita, daftar_klasifier, profil_ternormalisasi
from uji_kelayakan import kelompok_hampir_identik

KELUARAN = "uji_kekokohan.json"
PUSTAKA_KOMPRESI = f"OpenCV {cv2.__version__}"
SISI_KOMPRES = 1600
KUALITAS_JPEG = (85, 70, 50)
SKALA_INTENSITAS = (0.8, 1.2)
VARIASI_ROI = {"ujung 5%": {"tambahan_ujung": 0.05}, "ujung 15%": {"tambahan_ujung": 0.15},
               "lintang 5-95": {"persentil_lintang": (5, 95)}, "lintang 20-80": {"persentil_lintang": (20, 80)}}
JUMLAH_LAJUR = 4


def kompres(bgr, sisi, kualitas):
    skala = sisi / max(bgr.shape[:2])
    kecil = cv2.resize(bgr, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA) if skala < 1 else bgr
    return cv2.imdecode(cv2.imencode(".jpg", kecil, [cv2.IMWRITE_JPEG_QUALITY, kualitas])[1], cv2.IMREAD_COLOR)


def intensitas(bgr, faktor):
    return np.clip(bgr.astype(float) * faktor, 0, 255).astype(np.uint8)


def profil_dengan(bgr, setelan=None):
    return profil_ternormalisasi(ambil_pita(bgr, **(setelan or {}))[0])


def varian_f15(bgr):
    varian = {}
    for q in KUALITAS_JPEG:
        varian[f"kompres ulang {SISI_KOMPRES} px, JPEG {q}"] = ("kompresi", profil_dengan(kompres(bgr, SISI_KOMPRES, q)))
    for f in SKALA_INTENSITAS:
        varian[f"intensitas x{f:g}"] = ("intensitas", profil_dengan(intensitas(bgr, f)))
    for nama, setelan in VARIASI_ROI.items():
        varian[f"ROI {nama}"] = ("ROI", profil_dengan(bgr, setelan))
    return varian


def kumpulkan():
    baris = []
    for jalur in semua_citra():
        peran = peran_citra(jalur)
        if peran["peran"] != "foto spektrum":
            continue
        bgr = baca_bgr(jalur)
        pita, _ = ambil_pita(bgr)
        lebar = TITIK_LINTANG // JUMLAH_LAJUR
        baris.append({"jalur": jalur_relatif(jalur), "kelas": peran["kelas"], "lensa": peran["lensa"], "sisi_panjang_px": int(max(bgr.shape[:2])),
                      "profil": profil_ternormalisasi(pita),
                      "lajur": [profil_ternormalisasi(pita[:, i:i + lebar]) for i in range(0, TITIK_LINTANG, lebar)],
                      "varian": varian_f15(bgr) if peran["lensa"] == "f15" else {}})
    return baris


def latih_asli_uji_varian(klasifier, X, label, indeks, profil_varian):
    tebakan = []
    for i, profil in zip(indeks, profil_varian):
        latih = np.arange(len(X)) != i
        tebakan.append(klasifier.fit(X[latih], label[latih]).predict(profil[None])[0])
    return np.array(tebakan)


def loo_dengan_varian(klasifier, X, y, indeks, profil_varian):
    X_ganti = X.copy()
    X_ganti[indeks] = profil_varian
    return sum(int(klasifier.fit(X_ganti[np.arange(len(X)) != i], y[np.arange(len(X)) != i]).predict(X_ganti[i][None])[0] == y[i])
               for i in indeks)


def u7(baris):
    X = np.array([b["profil"] for b in baris])
    y = np.array([b["kelas"] for b in baris])
    lensa = np.array([b["lensa"] for b in baris])
    indeks = [i for i, b in enumerate(baris) if b["lensa"] == "f15"]
    hasil = {}
    for nama, (jenis, _) in baris[indeks[0]]["varian"].items():
        profil_varian = np.array([baris[i]["varian"][nama][1] for i in indeks])
        r = [float(np.corrcoef(X[i], v)[0, 1]) for i, v in zip(indeks, profil_varian)]
        per_klasifier = {}
        for nama_klasifier, klasifier in daftar_klasifier().items():
            isi = {"latih_asli_uji_varian_benar": int(np.sum(latih_asli_uji_varian(klasifier, X, y, indeks, profil_varian) == y[indeks])),
                   "semua_varian_loo_benar": loo_dengan_varian(klasifier, X, y, indeks, profil_varian), "n": len(indeks)}
            if jenis == "kompresi":
                tebak = latih_asli_uji_varian(klasifier, X, lensa, indeks, profil_varian)
                isi["lensa_tertebak"] = {l: int(np.sum(tebak == l)) for l in sorted(set(lensa))}
            per_klasifier[nama_klasifier] = isi
        hasil[nama] = {"jenis": jenis, "korelasi_profil_median": float(np.median(r)), "korelasi_profil_min": float(min(r)),
                       "klasifikasi": per_klasifier}
    lensa_asli = {nama: int(np.sum(latih_asli_uji_varian(k, X, lensa, indeks, X[indeks]) == "f15")) for nama, k in daftar_klasifier().items()}
    return {"pustaka_kompresi": PUSTAKA_KOMPRESI, "sisi_kompres_px": SISI_KOMPRES, "kualitas_jpeg": KUALITAS_JPEG,
            "sisi_panjang_px_per_lensa": {l: sorted({b["sisi_panjang_px"] for b in baris if b["lensa"] == l}) for l in sorted(set(lensa))},
            "n_f15": len(indeks), "lensa_f15_asli_tertebak_f15": lensa_asli, "varian": hasil}


def tebak_per_foto(klasifier, Xp, yp, gp, kelompok_uji):
    tebakan = {}
    for latih, uji in kelompok_uji:
        klasifier.fit(Xp[latih], yp[latih])
        for foto in np.unique(gp[uji]):
            suara = klasifier.predict(Xp[uji][gp[uji] == foto])
            nilai, jumlah = np.unique(suara, return_counts=True)
            tebakan[foto] = nilai[np.argmax(jumlah)]
    return tebakan


def u8(baris):
    y_foto = np.array([b["kelas"] for b in baris])
    kelompok_foto = kelompok_hampir_identik([{"jalur": b["jalur"]} for b in baris])
    Xp = np.array([l for b in baris for l in b["lajur"]])
    yp = np.repeat(y_foto, JUMLAH_LAJUR)
    gp = np.repeat(np.arange(len(baris)), JUMLAH_LAJUR)
    kp = np.repeat(kelompok_foto, JUMLAH_LAJUR)
    hasil = {}
    for nama_klasifier, klasifier in daftar_klasifier().items():
        per = {}
        for nama, grup in (("tinggalkan satu foto", gp), ("tinggalkan satu kelompok hampir identik", kp)):
            tebakan = tebak_per_foto(klasifier, Xp, yp, gp, LeaveOneGroupOut().split(Xp, yp, grup))
            benar_patch = sum(int(np.sum(klasifier.fit(Xp[latih], yp[latih]).predict(Xp[uji]) == yp[uji]))
                              for latih, uji in LeaveOneGroupOut().split(Xp, yp, grup))
            per[nama] = {"benar_per_foto": int(sum(tebakan[i] == y_foto[i] for i in range(len(baris)))), "n_foto": len(baris),
                         "benar_per_patch": benar_patch, "n_patch": len(yp)}
        hasil[nama_klasifier] = per
    return {"jumlah_lajur": JUMLAH_LAJUR, "titik_lintang_per_lajur": TITIK_LINTANG // JUMLAH_LAJUR,
            "jumlah_kelompok_hampir_identik": int(len(set(kelompok_foto))), "hasil": hasil}


def main():
    baris = kumpulkan()
    tulis_json(KELUARAN, {"titik_sumbu": TITIK_SUMBU, "titik_lintang": TITIK_LINTANG, "u7": u7(baris), "u8": u8(baris)})


if __name__ == "__main__":
    main()
