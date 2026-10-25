import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from gambar_bersama import (DUA_KOLOM, FOTO_CONTOH, KERTAS, LABEL_PERSEN, LABEL_POSISI, TINTA,
                            gambar_grid, gambar_profil, gambar_strip, kotak_potong, langkah_pita, simpan, tampil_citra)
from model import TITIK_LINTANG, TITIK_SUMBU

NAMA = "Figure_1b"
PERSENTIL_TAMPIL = 99.5


def main():
    langkah = langkah_pita(FOTO_CONTOH)
    potong = kotak_potong(langkah)
    gbr = plt.figure(figsize=(DUA_KOLOM, 1.95))
    kiri = gbr.add_gridspec(1, 4, left=0.03, right=0.6, top=0.86, bottom=0.06, wspace=0.08)
    kanan = gbr.add_gridspec(2, 1, height_ratios=[1, 3.2], left=0.675, right=0.955, top=0.86, bottom=0.2, hspace=0.3)
    foto = gbr.add_subplot(kiri[0])
    tampil_citra(foto, langkah["rgb"])
    foto.add_patch(Rectangle(potong[:2], potong[2], potong[3], fill=False, ec=KERTAS, lw=0.7, ls=(0, (3, 2))))
    foto.set_title("Photograph")
    ax = gbr.add_subplot(kiri[1])
    tampil_citra(ax, langkah["kroma"], potong, cmap="Greys", vmax=np.percentile(langkah["kroma"], PERSENTIL_TAMPIL), bingkai=True)
    ax.set_title("Chroma")
    ax = gbr.add_subplot(kiri[2])
    tampil_citra(ax, langkah["topeng"], potong, cmap="Greys", vmin=0, vmax=2.2, bingkai=True)
    ax.contour(langkah["topeng"].astype(float), levels=[0.5], colors=TINTA, linewidths=0.7)
    ax.set_xlim(potong[0], potong[0] + potong[2])
    ax.set_ylim(potong[1] + potong[3], potong[1])
    ax.set_title("Band mask")
    ax = gbr.add_subplot(kiri[3])
    tampil_citra(ax, langkah["rgb"], potong)
    gambar_grid(ax, langkah["geometri"])
    ax.set_xlim(potong[0], potong[0] + potong[2])
    ax.set_ylim(potong[1] + potong[3], potong[1])
    ax.set_title("Axis and grid")
    strip = gbr.add_subplot(kanan[0])
    gambar_strip(strip, langkah["pita"])
    strip.set_title(f"{TITIK_SUMBU} × {TITIK_LINTANG} grid samples")
    profil = gbr.add_subplot(kanan[1])
    gambar_profil(profil, langkah["profil"])
    profil.set_xlabel(LABEL_POSISI, labelpad=1.5)
    profil.set_ylabel(LABEL_PERSEN, labelpad=2)
    profil.set_title("Normalized R, G, B profile")
    gbr.text(0.0, 0.955, "(b)", fontsize=9, fontweight="bold", va="top")
    simpan(gbr, NAMA)


if __name__ == "__main__":
    main()
