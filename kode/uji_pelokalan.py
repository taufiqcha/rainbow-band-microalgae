import cv2
import numpy as np

from data import baca_bgr, foto_berlabel, jalur_relatif, simpan_gambar, tulis_json
from gambar_bersama import AKSEN, KERTAS, TINTA, plt
from model import (ALFA_RONA, BATAS_KROMA, FRAKSI_KROMA, KONSISTENSI_NORMAL, MAYORITAS, PERSENTIL_KROMA, TAMBAHAN_UJUNG, ambil_pita,
                   batas_sisa_pita, fraksi_jenuh, garis_lms, geometri_pita, kroma, mutu_pelokalan, titik_grid, titik_inti,
                   topeng_pelangi)

KELUARAN = "uji_pelokalan.json"
NAMA_GAMBAR = "e2_3_tumpang_pita"
LENSA = ("f15", "f20")
KOLOM = 6
PARAMETER_GARIS = 2
SUKU_SAMPEL_KECIL = 5
SISI_TAMPIL = 256
SETENGAH_POTONGAN = 0.75


def ukur_foto(jalur, peran):
    bgr = baca_bgr(jalur)
    kecil, skala, geometri = geometri_pita(bgr)
    pita, info = ambil_pita(bgr, geometri=geometri)
    suku_fraksi = float(FRAKSI_KROMA * np.percentile(kroma(kecil), PERSENTIL_KROMA))
    panjang_warna = info["panjang_px"] / (1 + 2 * TAMBAHAN_UJUNG)
    ukuran = sorted(bgr.shape[:2], reverse=True)
    baris = {"jalur": jalur_relatif(jalur), "kelas": peran["kelas"], "lensa": peran["lensa"],
             "ukuran_citra_px": f"{ukuran[0]} x {ukuran[1]}", "panjang_pita_berwarna_px": panjang_warna,
             "panjang_relatif_sisi_panjang": panjang_warna / ukuran[0], "lebar_pita_px": info["lebar_px"],
             "sudut_sumbu_derajat": info["sudut_sumbu_derajat"], "ambang_kroma": max(BATAS_KROMA, suku_fraksi),
             "suku_ambang_aktif": "BATAS_KROMA" if BATAS_KROMA >= suku_fraksi else "FRAKSI_KROMA",
             "fraksi_jenuh_pita": fraksi_jenuh(pita), **mutu_pelokalan(kecil, geometri, pita), **titik_inti_dan_koreksi(kecil)}
    return baris, {"kecil": kecil, "topeng": topeng_pelangi(kecil), "geometri": geometri}


def koreksi_sampel_kecil(n):
    return 1 + SUKU_SAMPEL_KECIL / (n - PARAMETER_GARIS)


def titik_inti_dan_koreksi(kecil):
    _, _, titik = titik_inti(topeng_pelangi(kecil), kecil)
    sisa = np.abs(garis_lms(titik))
    koreksi = koreksi_sampel_kecil(len(titik))
    return {"jumlah_titik_inti": len(titik), "faktor_koreksi_sampel_kecil": koreksi,
            "titik_pita_bertambah_dengan_koreksi": int(np.sum(sisa <= batas_sisa_pita(sisa, koreksi)) - np.sum(sisa <= batas_sisa_pita(sisa)))}


def ringkas_koreksi(baris):
    inti = [b["jumlah_titik_inti"] for b in baris]
    return {"rumus": f"{KONSISTENSI_NORMAL} x (1 + {SUKU_SAMPEL_KECIL}/(n - {PARAMETER_GARIS})) Rousseeuw-Leroy 1987 tidak dipakai; model memakai {KONSISTENSI_NORMAL} saja",
            "jumlah_titik_inti_min": min(inti), "jumlah_titik_inti_median": float(np.median(inti)), "jumlah_titik_inti_maks": max(inti),
            "faktor_koreksi_maks": koreksi_sampel_kecil(min(inti)),
            "titik_pita_bertambah_total": sum(b["titik_pita_bertambah_dengan_koreksi"] for b in baris)}


def hitung_ditemukan(baris):
    hasil = {}
    for b in baris:
        sel = hasil.setdefault(b["kelas"], {}).setdefault(b["lensa"], {"ditemukan": 0, "n": 0})
        sel["n"] += 1
        sel["ditemukan"] += int(b["ditemukan"])
    return {k: dict(sorted(v.items())) for k, v in sorted(hasil.items())}


def ringkas_panjang(baris, kunci):
    kelompok = {}
    for b in baris:
        kelompok.setdefault(kunci(b), []).append(b)
    return {k: {"n": len(v), "kelas": sorted({b["kelas"] for b in v}), "lensa": sorted({b["lensa"] for b in v}),
                "panjang_pita_berwarna_px_median": float(np.median([b["panjang_pita_berwarna_px"] for b in v])),
                "panjang_relatif_sisi_panjang_median": float(np.median([b["panjang_relatif_sisi_panjang"] for b in v]))}
            for k, v in sorted(kelompok.items())}


def ringkas_jenuh(baris):
    return {l: {"median": float(np.median([b["fraksi_jenuh_pita"] for b in baris if b["lensa"] == l])),
                "maks": max(b["fraksi_jenuh_pita"] for b in baris if b["lensa"] == l)} for l in LENSA}


def panel_tumpang(ax, nomor, baris, citra):
    kecil, geometri = citra["kecil"], citra["geometri"]
    pusat, arah, _, (awal, akhir), _ = geometri
    setengah = SETENGAH_POTONGAN * (akhir - awal)
    x0, x1 = int(max(0, pusat[0] - setengah)), int(min(kecil.shape[1], pusat[0] + setengah))
    y0, y1 = int(max(0, pusat[1] - setengah)), int(min(kecil.shape[0], pusat[1] + setengah))
    skala = SISI_TAMPIL / max(x1 - x0, y1 - y0)
    potongan = cv2.resize(kecil[y0:y1, x0:x1], None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
    ax.imshow(potongan[:, :, ::-1], extent=(x0, x1, y1, y0), interpolation="nearest")
    ax.contour(np.arange(x0, x1), np.arange(y0, y1), citra["topeng"][y0:y1, x0:x1].astype(float), levels=[0.5],
               colors=KERTAS, linewidths=0.5)
    sudut_x, sudut_y = titik_grid(geometri, 2, 2)
    ax.plot(np.r_[sudut_x[0], sudut_x[1, ::-1], sudut_x[0, 0]], np.r_[sudut_y[0], sudut_y[1, ::-1], sudut_y[0, 0]], color=AKSEN, lw=0.8)
    poros = pusat + np.outer((awal, akhir), arah)
    ax.plot(poros[:, 0], poros[:, 1], color=AKSEN, lw=0.6, ls=(0, (3, 2)))
    ax.plot(*poros[0], "o", color=AKSEN, ms=2.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sisi in ax.spines.values():
        sisi.set_visible(True)
        sisi.set_color(TINTA)
        sisi.set_linewidth(0.5)
    ax.set_title(f"{nomor}. {baris['kelas']} {baris['lensa']}", fontsize=7, color=TINTA, pad=2)


def gambar_tumpang(semua):
    baris_panel = -(-len(semua) // KOLOM)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    gbr, sumbu = plt.subplots(baris_panel, KOLOM, figsize=(7.48, 1.3 * baris_panel))
    for i, ax in enumerate(sumbu.ravel()):
        if i < len(semua):
            panel_tumpang(ax, i + 1, *semua[i])
        else:
            ax.axis("off")
    gbr.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.06, hspace=0.22)
    alamat = simpan_gambar(gbr, NAMA_GAMBAR)
    plt.close(gbr)
    return alamat


def main():
    semua = [ukur_foto(jalur, peran) for jalur, peran in foto_berlabel()]
    baris = [b for b, _ in semua]
    tulis_json(KELUARAN, {
        "kriteria_ditemukan": (f"seluruh grid di dalam bingkai citra, fraksi titik sumbu yang lajurnya mayoritas di topeng > {MAYORITAS}, "
                               f"dan rona naik sepanjang sumbu (Spearman rho > 0, p < {ALFA_RONA})"),
        "n": len(baris), "ditemukan": sum(b["ditemukan"] for b in baris), "ditemukan_per_kelas_lensa": hitung_ditemukan(baris),
        "fraksi_sumbu_dalam_topeng_min": min(b["fraksi_sumbu_dalam_topeng"] for b in baris),
        "fraksi_sumbu_dalam_topeng_maks": max(b["fraksi_sumbu_dalam_topeng"] for b in baris),
        "spearman_rona_sumbu_min": min(b["spearman_rona_sumbu"] for b in baris),
        "koreksi_sampel_kecil_lms": ringkas_koreksi(baris),
        "p_spearman_rona_sumbu_maks": max(b["p_spearman_rona_sumbu"] for b in baris),
        "suku_ambang_aktif": {s: sum(b["suku_ambang_aktif"] == s for b in baris) for s in ("BATAS_KROMA", "FRAKSI_KROMA")},
        "panjang_per_lensa": ringkas_panjang(baris, lambda b: b["lensa"]),
        "panjang_per_ukuran": ringkas_panjang(baris, lambda b: b["ukuran_citra_px"]),
        "panjang_per_lensa_ukuran": ringkas_panjang(baris, lambda b: f"{b['lensa']} | {b['ukuran_citra_px']}"),
        "fraksi_jenuh_per_lensa": ringkas_jenuh(baris), "gambar": gambar_tumpang(semua),
        "citra": [{"nomor_panel": i + 1, **b} for i, b in enumerate(baris)]})


if __name__ == "__main__":
    main()
