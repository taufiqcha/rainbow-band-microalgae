import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import SEED

SISI_KERJA = 1200
BATAS_KROMA = 30
FRAKSI_KROMA = 0.35
PERSENTIL_KROMA = 99
PERSENTIL_INTI = 60
TITIK_SUMBU = 64
TITIK_LINTANG = 16
PERSENTIL_SUMBU = (1, 99)
PERSENTIL_LINTANG = (10, 90)
BATAS_RONA_MERAH = 150
TAMBAHAN_UJUNG = 0.10
BATAS_JENUH = 254
JUMLAH_LIPAT = 5
ULANGAN_LIPAT = 20
JUMLAH_PERMUTASI = 1000
ALFA_RONA = 0.05
LANGKAH_SUDUT_LMS = 1.0
KONSISTENSI_NORMAL = 1.4826
BATAS_SISA_BAKU = 2.5
MAYORITAS = 0.5


def perkecil(bgr):
    skala = SISI_KERJA / max(bgr.shape[:2])
    return cv2.resize(bgr, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA), skala


def kroma(bgr):
    halus = cv2.GaussianBlur(bgr, (9, 9), 0).astype(int)
    return halus.max(axis=2) - halus.min(axis=2)


def topeng_pelangi(bgr, batas_kroma=BATAS_KROMA, fraksi_kroma=FRAKSI_KROMA):
    nilai = kroma(bgr)
    topeng = (nilai >= max(batas_kroma, fraksi_kroma * np.percentile(nilai, PERSENTIL_KROMA))).astype(np.uint8)
    topeng = cv2.morphologyEx(topeng, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    topeng = cv2.morphologyEx(topeng, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    _, label, statistik, _ = cv2.connectedComponentsWithStats(topeng)
    return label == 1 + int(np.argmax(statistik[1:, cv2.CC_STAT_AREA]))


def rona_bersambung(hsv):
    rona = hsv[:, :, 0].astype(float)
    return np.where(rona > BATAS_RONA_MERAH, rona - 180, rona)


def garis_lms(titik):
    sudut = np.radians(np.arange(0, 180, LANGKAH_SUDUT_LMS))
    normal = np.column_stack([np.cos(sudut), np.sin(sudut)])
    jarak = np.sort(titik @ normal.T, axis=0)
    separuh = len(titik) // 2 + 1
    lebar = jarak[separuh - 1:] - jarak[:len(titik) - separuh + 1]
    awal = np.argmin(lebar, axis=0)
    terbaik = int(np.argmin(lebar[awal, np.arange(len(sudut))]))
    tengah = (jarak[awal[terbaik], terbaik] + jarak[awal[terbaik] + separuh - 1, terbaik]) / 2
    return titik @ normal[terbaik] - tengah


def titik_inti(topeng, bgr, persentil_inti=PERSENTIL_INTI):
    nilai = kroma(bgr)
    y, x = np.nonzero(topeng & (nilai >= np.percentile(nilai[topeng], persentil_inti)))
    return y, x, np.column_stack([x, y]).astype(float)


def batas_sisa_pita(sisa, koreksi=1.0):
    return BATAS_SISA_BAKU * KONSISTENSI_NORMAL * koreksi * np.median(np.abs(sisa))


def sumbu_dispersi(topeng, bgr, tambahan_ujung=TAMBAHAN_UJUNG, persentil_lintang=PERSENTIL_LINTANG, persentil_inti=PERSENTIL_INTI):
    y, x, titik = titik_inti(topeng, bgr, persentil_inti)
    sisa = garis_lms(titik)
    pita = np.abs(sisa) <= batas_sisa_pita(sisa)
    titik, y, x = titik[pita], y[pita], x[pita]
    pusat = titik.mean(axis=0)
    _, _, vt = np.linalg.svd(titik - pusat, full_matrices=False)
    arah = vt[0]
    rona = rona_bersambung(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))[y, x]
    if np.corrcoef((titik - pusat) @ arah, rona)[0, 1] < 0:
        arah = -arah
    lintang = np.array([-arah[1], arah[0]])
    ty, tx = np.nonzero(topeng)
    semua = np.column_stack([tx, ty]).astype(float) - pusat
    awal, akhir = np.percentile(semua @ arah, PERSENTIL_SUMBU)
    tambahan = tambahan_ujung * (akhir - awal)
    return pusat, arah, lintang, (awal - tambahan, akhir + tambahan), tuple(np.percentile((titik - pusat) @ lintang, persentil_lintang))


def geometri_kecil(kecil, tambahan_ujung=TAMBAHAN_UJUNG, persentil_lintang=PERSENTIL_LINTANG, batas_kroma=BATAS_KROMA,
                   fraksi_kroma=FRAKSI_KROMA, persentil_inti=PERSENTIL_INTI):
    return sumbu_dispersi(topeng_pelangi(kecil, batas_kroma, fraksi_kroma), kecil, tambahan_ujung, persentil_lintang, persentil_inti)


def titik_grid(geometri, titik=TITIK_SUMBU, titik_lintang=TITIK_LINTANG):
    pusat, arah, lintang, (awal, akhir), (kiri, kanan) = geometri
    grid_s, grid_t = np.meshgrid(np.linspace(awal, akhir, titik), np.linspace(kiri, kanan, titik_lintang), indexing="ij")
    return pusat[0] + grid_s * arah[0] + grid_t * lintang[0], pusat[1] + grid_s * arah[1] + grid_t * lintang[1]


def ambil_pita(bgr, titik=TITIK_SUMBU, geometri=None, titik_lintang=TITIK_LINTANG, **pelokalan):
    kecil, skala = perkecil(bgr)
    geometri = geometri or geometri_kecil(kecil, **pelokalan)
    _, arah, _, (awal, akhir), (kiri, kanan) = geometri
    grid_x, grid_y = titik_grid(geometri, titik, titik_lintang)
    pita = cv2.remap(bgr.astype(np.float32), (grid_x / skala).astype(np.float32), (grid_y / skala).astype(np.float32), cv2.INTER_LINEAR)
    return pita[:, :, ::-1], {"panjang_px": float((akhir - awal) / skala), "lebar_px": float((kanan - kiri) / skala),
                              "sudut_sumbu_derajat": float(np.degrees(np.arctan2(arah[1], arah[0])))}


def geometri_pita(bgr, **pelokalan):
    kecil, skala = perkecil(bgr)
    return kecil, skala, geometri_kecil(kecil, **pelokalan)


def fraksi_sumbu_dalam_topeng(geometri, topeng, titik=TITIK_SUMBU, titik_lintang=TITIK_LINTANG):
    grid_x, grid_y = titik_grid(geometri, titik, titik_lintang)
    kolom = np.clip(np.round(grid_x).astype(int), 0, topeng.shape[1] - 1)
    baris = np.clip(np.round(grid_y).astype(int), 0, topeng.shape[0] - 1)
    return float(np.mean(topeng[baris, kolom].mean(axis=1) > MAYORITAS))


def mutu_pelokalan(kecil, geometri, pita_rgb, batas_kroma=BATAS_KROMA, fraksi_kroma=FRAKSI_KROMA):
    grid_x, grid_y = titik_grid(geometri, *pita_rgb.shape[:2])
    tinggi, lebar = kecil.shape[:2]
    dalam = float(np.mean((grid_x >= 0) & (grid_x <= lebar - 1) & (grid_y >= 0) & (grid_y <= tinggi - 1)))
    sumbu_topeng = fraksi_sumbu_dalam_topeng(geometri, topeng_pelangi(kecil, batas_kroma, fraksi_kroma), *pita_rgb.shape[:2])
    hsv = cv2.cvtColor(np.clip(pita_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    rho, p = spearmanr(np.arange(len(pita_rgb)), rona_bersambung(hsv).mean(axis=1))
    return {"fraksi_grid_dalam_bingkai": dalam, "fraksi_sumbu_dalam_topeng": sumbu_topeng, "spearman_rona_sumbu": float(rho),
            "p_spearman_rona_sumbu": float(p),
            "ditemukan": bool(dalam == 1 and sumbu_topeng > MAYORITAS and rho > 0 and p < ALFA_RONA)}



def profil_ternormalisasi(pita_rgb):
    profil = pita_rgb.mean(axis=1)
    return (profil / profil.sum()).T.ravel()


def fitur_rerata_ta(pita_rgb):
    rgb = np.clip(pita_rgb, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).mean(axis=0)
    return np.concatenate([rgb.reshape(-1, 3).mean(axis=0), hsv])


def fraksi_jenuh(pita_rgb):
    return float(np.mean(pita_rgb.max(axis=2) >= BATAS_JENUH))


def fitur_latar(bgr):
    kecil, _ = perkecil(bgr)
    topeng = cv2.dilate(topeng_pelangi(kecil).astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
    latar = kecil[~topeng].astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(latar, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    return np.concatenate([latar.reshape(-1, 3)[:, ::-1].mean(axis=0), hsv])


def daftar_klasifier():
    return {"1-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
            "regresi logistik L2": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))}


def prediksi_loo(klasifier, X, y):
    tebakan = np.empty_like(y)
    for latih, uji in LeaveOneOut().split(X):
        tebakan[uji] = klasifier.fit(X[latih], y[latih]).predict(X[uji])
    return tebakan


def akurasi_5lipat_berulang(klasifier, X, y):
    hasil = []
    for ulang in range(ULANGAN_LIPAT):
        tebakan = np.empty_like(y)
        for latih, uji in StratifiedKFold(JUMLAH_LIPAT, shuffle=True, random_state=SEED + ulang).split(X, y):
            tebakan[uji] = klasifier.fit(X[latih], y[latih]).predict(X[uji])
        hasil.append(float(np.mean(tebakan == y)))
    return hasil


def akurasi_loo_teracak(klasifier, X, y, benih):
    y_acak = np.random.default_rng(benih).permutation(y)
    return float(np.mean(prediksi_loo(klasifier, X, y_acak) == y_acak))


def p_permutasi_loo(klasifier, X, y, akurasi_amatan):
    acak = Parallel(n_jobs=-1)(delayed(akurasi_loo_teracak)(clone(klasifier), X, y, SEED + i) for i in range(JUMLAH_PERMUTASI))
    return (sum(a >= akurasi_amatan for a in acak) + 1) / (JUMLAH_PERMUTASI + 1)


def selang_wilson(benar, total, z=1.959963984540054):
    p = benar / total
    pusat = (p + z * z / (2 * total)) / (1 + z * z / total)
    lebar = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return float(max(0.0, pusat - lebar)), float(min(1.0, pusat + lebar))


def histogram_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return [cv2.calcHist([hsv], [0], None, [180], [0, 180]), cv2.calcHist([hsv], [1], None, [256], [0, 256]),
            cv2.calcHist([hsv], [2], None, [256], [0, 256])]


def interseksi_ta(hist_a, hist_b):
    return float(np.mean([np.sum(np.minimum(a, b)) / np.sum(b) for a, b in zip(hist_a, hist_b)]))


def akurasi_1nn_kemiripan(kemiripan, y):
    salinan = kemiripan.copy()
    np.fill_diagonal(salinan, -np.inf)
    return float(np.mean(y[np.argmax(salinan, axis=1)] == y))


def p_permutasi_kemiripan(kemiripan, y, akurasi_amatan):
    acak = np.random.default_rng(SEED)
    lebih = sum(akurasi_1nn_kemiripan(kemiripan, y_acak) >= akurasi_amatan for y_acak in (acak.permutation(y) for _ in range(JUMLAH_PERMUTASI)))
    return (lebih + 1) / (JUMLAH_PERMUTASI + 1)


def akurasi_per_kelompok(klasifier, X, y, kelompok):
    tebakan = np.empty_like(y)
    for latih, uji in LeaveOneGroupOut().split(X, y, kelompok):
        tebakan[uji] = klasifier.fit(X[latih], y[latih]).predict(X[uji])
    return int(np.sum(tebakan == y))
