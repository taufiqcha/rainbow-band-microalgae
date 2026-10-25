import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import AKAR, TA, baca_bgr
from model import (TITIK_LINTANG, TITIK_SUMBU, ambil_pita, geometri_pita, kroma, profil_ternormalisasi, titik_grid,
                   topeng_pelangi)

FIGUR = AKAR / "figures"
DUA_KOLOM = 7.16
TINTA = "#2d3142"
REDUP = "#4f5d75"
LUNAK = "#5a6580"
GARIS = "#bfc0c0"
AKSEN = "#bf4520"
KERTAS = "#ffffff"
GAYA_KELAS = {"Chlorella": {"color": TINTA, "ls": "-"},
              "Nannochloropsis": {"color": REDUP, "ls": "--"},
              "Spirulina": {"color": LUNAK, "ls": ":"}}
GAYA_KANAL = {"R": {"color": TINTA, "ls": "-"}, "G": {"color": REDUP, "ls": "--"}, "B": {"color": LUNAK, "ls": ":"}}
FOTO_CONTOH = TA / "Testing" / "F.15" / "Chorella" / "Chorella 1.jpg"
LABEL_POSISI = "Relative position (0 = red end, 1 = blue end)"
LABEL_PERSEN = "% of profile sum"
LONGGAR_POTONG = 0.45
RASIO_POTONG = 4 / 3
LANGKAH_GRID = 8
RUANG_ATAS = 1.22
GESER_UJUNG = 0.6

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom", "mathtext.it": "Arial:italic", "mathtext.rm": "Arial",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": TINTA, "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": TINTA, "ytick.color": TINTA, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GARIS, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlepad": 4,
    "pdf.fonttype": 42, "figure.facecolor": KERTAS, "axes.facecolor": KERTAS, "savefig.facecolor": KERTAS})


def miring(nama):
    return rf"$\it{{{nama}}}$"


def simpan(gbr, nama, lebar_png=None):
    FIGUR.mkdir(exist_ok=True)
    gbr.savefig(FIGUR / f"{nama}.pdf", metadata={"CreationDate": None, "ModDate": None})
    if lebar_png:
        gbr.savefig(FIGUR / f"{nama}.png", dpi=lebar_png / gbr.get_figwidth(), metadata={"Software": None})
    plt.close(gbr)
    return f"figures/{nama}.pdf"


def langkah_pita(jalur):
    bgr = baca_bgr(jalur)
    kecil, _, geometri = geometri_pita(bgr)
    pita, ukuran = ambil_pita(bgr, geometri=geometri)
    return {"rgb": kecil[:, :, ::-1], "kroma": kroma(kecil), "topeng": topeng_pelangi(kecil), "geometri": geometri,
            "pita": pita, "profil": profil_ternormalisasi(pita), "ukuran": ukuran}


def kotak_potong(langkah):
    x, y = titik_grid(langkah["geometri"])
    ty, tx = np.nonzero(langkah["topeng"])
    x0, x1 = min(x.min(), tx.min()), max(x.max(), tx.max())
    y0, y1 = min(y.min(), ty.min()), max(y.max(), ty.max())
    tinggi = max(y1 - y0, (x1 - x0) * RASIO_POTONG) * (1 + 2 * LONGGAR_POTONG)
    lebar = tinggi / RASIO_POTONG
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    batas_y, batas_x = langkah["topeng"].shape
    kiri = int(np.clip(cx - lebar / 2, 0, batas_x - lebar))
    atas = int(np.clip(cy - tinggi / 2, 0, batas_y - tinggi))
    return kiri, atas, int(lebar), int(tinggi)


def tampil_citra(ax, citra, potong=None, bingkai=False, **kw):
    if potong:
        kiri, atas, lebar, tinggi = potong
        citra = citra[atas:atas + tinggi, kiri:kiri + lebar]
        ax.imshow(citra, extent=(kiri, kiri + lebar, atas + tinggi, atas), interpolation="antialiased", **kw)
    else:
        ax.imshow(citra, interpolation="antialiased", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_anchor("N")
    for sisi in ax.spines.values():
        sisi.set_visible(bingkai)
        sisi.set_color(GARIS)
        sisi.set_linewidth(0.6)


def gambar_grid(ax, geometri, warna_grid=KERTAS, lebar=0.6, ujung=True):
    x, y = titik_grid(geometri)
    for i in list(range(0, TITIK_SUMBU, LANGKAH_GRID)) + [TITIK_SUMBU - 1]:
        ax.plot(x[i, [0, -1]], y[i, [0, -1]], color=warna_grid, lw=lebar)
    for j in (0, TITIK_LINTANG - 1):
        ax.plot(x[[0, -1], j], y[[0, -1], j], color=warna_grid, lw=lebar)
    tx, ty = x.mean(axis=1), y.mean(axis=1)
    ax.annotate("", xy=(tx[-1], ty[-1]), xytext=(tx[0], ty[0]),
                arrowprops={"arrowstyle": "-|>", "color": AKSEN, "lw": 1.3, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7})
    if ujung:
        for i, teks in ((0, "0"), (-1, "1")):
            ax.text(x[i, -1] + GESER_UJUNG * (x[i, -1] - x[i, 0]), y[i, -1] + GESER_UJUNG * (y[i, -1] - y[i, 0]), teks,
                    color=warna_grid, fontsize=7, ha="center", va="center")


def gambar_strip(ax, pita):
    ax.imshow(np.clip(pita.transpose(1, 0, 2) / 255, 0, 1), extent=(0, 1, 0, 1), aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sisi in ax.spines.values():
        sisi.set_visible(False)


def gambar_profil(ax, profil, label_puncak=True, lebar=1.1):
    posisi = np.linspace(0, 1, TITIK_SUMBU)
    for i, (kanal, gaya) in enumerate(GAYA_KANAL.items()):
        nilai = profil[i * TITIK_SUMBU:(i + 1) * TITIK_SUMBU] * 100
        ax.plot(posisi, nilai, lw=lebar, **gaya)
        if label_puncak:
            puncak = int(np.argmax(nilai))
            ax.annotate(kanal, (posisi[puncak], nilai[puncak]), xytext=(0, 2), textcoords="offset points",
                        color=gaya["color"], fontsize=7.5, ha="center", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, RUANG_ATAS * profil.max() * 100)
    ax.set_xticks([0, 0.5, 1])
