import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import f as sebaran_f
from statsmodels.stats.multitest import multipletests

from data import baca_json
from gambar_bersama import DUA_KOLOM, GAYA_KELAS, LABEL_PERSEN, LABEL_POSISI, TINTA, miring, simpan

NAMA = "Figure_2"
LENSA = {"f15": "f = 15 cm lens", "f20": "f = 20 cm lens"}
GAYA_F = {"color": TINTA, "ls": "-"}


def gabung(kelompok):
    n = sum(k["n"] for k in kelompok)
    rerata = sum(k["n"] * k["rerata"] for k in kelompok) / n
    jk = sum((k["n"] - 1) * k["sd"] ** 2 + k["n"] * (k["rerata"] - rerata) ** 2 for k in kelompok)
    return {"n": n, "rerata": rerata, "jk": jk}


def anova_dari_ringkasan(kelas):
    n = sum(k["n"] for k in kelas)
    rerata = sum(k["n"] * k["rerata"] for k in kelas) / n
    antar = sum(k["n"] * (k["rerata"] - rerata) ** 2 for k in kelas) / (len(kelas) - 1)
    dalam = sum(k["jk"] for k in kelas) / (n - len(kelas))
    f = antar / dalam
    q = multipletests(sebaran_f.sf(f, len(kelas) - 1, n - len(kelas)), method="fdr_bh")[1]
    return f, q, n


def ringkasan_kelompok(u1):
    hasil = {}
    for kunci, isi in u1["profil_kelas"].items():
        kelas, lensa = kunci.split(" | ")
        hasil[(kelas, lensa)] = {"n": isi["n"], "rerata": np.array(isi["rerata"]), "sd": np.array(isi["sd"])}
    return hasil


def statistik_f(kelompok, kelas, lensa_dipakai):
    return anova_dari_ringkasan([gabung([kelompok[(k, l)] for l in lensa_dipakai]) for k in kelas])


def cocokkan(f, q, ringkas, titik, batas_q):
    for i, kanal in enumerate("RGB"):
        bagian = slice(i * titik, (i + 1) * titik)
        assert np.isclose(f[bagian].max(), ringkas[kanal]["F_maks"])
        assert int(np.argmax(f[bagian])) == ringkas[kanal]["titik_F_maks"]
        assert int(np.sum(q[bagian] < batas_q)) == ringkas[kanal]["jumlah_titik_q_kurang_005"]


def main():
    sumber = baca_json("uji_pembeda.json")
    u1, titik = sumber["u1"], sumber["titik_sumbu"]
    kelompok = ringkasan_kelompok(u1)
    kelas = list(GAYA_KELAS)
    f15, q15, n15 = statistik_f(kelompok, kelas, ("f15",))
    cocokkan(f15, q15, u1["F_kelas_f15_saja"], titik, u1["batas_q_bh"])
    posisi = np.linspace(0, 1, titik)
    gbr, sumbu = plt.subplots(3, 3, figsize=(DUA_KOLOM, 5.0), sharex=True,
                              gridspec_kw={"left": 0.075, "right": 0.99, "top": 0.95, "bottom": 0.2, "wspace": 0.22, "hspace": 0.18})
    for i, kanal in enumerate("RGB"):
        bagian = slice(i * titik, (i + 1) * titik)
        for kolom, lensa in enumerate(LENSA):
            ax = sumbu[i, kolom]
            for k in kelas:
                isi = kelompok[(k, lensa)]
                rerata, sd = isi["rerata"][bagian] * 100, isi["sd"][bagian] * 100
                ax.fill_between(posisi, rerata - sd, rerata + sd, color=GAYA_KELAS[k]["color"], alpha=0.14, lw=0)
                ax.plot(posisi, rerata, lw=1.1, **GAYA_KELAS[k])
        bawah = min(sumbu[i, 0].get_ylim()[0], sumbu[i, 1].get_ylim()[0], 0)
        atas = max(sumbu[i, 0].get_ylim()[1], sumbu[i, 1].get_ylim()[1])
        for kolom in (0, 1):
            sumbu[i, kolom].set_ylim(bawah, atas)
        sumbu[i, 0].set_ylabel(f"{kanal} ({LABEL_PERSEN})")
        ax = sumbu[i, 2]
        batas = 1.08 * f15[bagian].max()
        ax.plot(posisi, f15[bagian], lw=1.1, **GAYA_F)
        bermakna = q15[bagian] < u1["batas_q_bh"]
        ax.plot(posisi[bermakna], np.full(bermakna.sum(), -0.06 * batas), "|", color=GAYA_F["color"], ms=3, mew=0.8)
        ax.set_ylim(-0.12 * batas, batas)
    for kolom, lensa in enumerate(LENSA):
        n = [kelompok[(k, lensa)]["n"] for k in kelas]
        sumbu[0, kolom].set_title(LENSA[lensa])
        sumbu[2, kolom].legend([Line2D([], [], lw=1.1, **GAYA_KELAS[k]) for k in kelas],
                               [f"{miring(k)}, n = {m}" for k, m in zip(kelas, n)],
                               loc="upper left", bbox_to_anchor=(0, -0.3), frameon=False, handlelength=2.4)
    sumbu[0, 2].set_title("F statistic across species, f = 15 cm")
    sumbu[2, 2].legend([Line2D([], [], lw=1.1, **GAYA_F), Line2D([], [], color=TINTA, marker="|", ls="none", ms=5, mew=0.8)],
                       [f"F, f = 15 cm lens, n = {n15}", f"BH q < {u1['batas_q_bh']}"],
                       loc="upper left", bbox_to_anchor=(0, -0.3), frameon=False, handlelength=2.4)
    for ax in sumbu[2]:
        ax.set_xlabel(LABEL_POSISI if ax is sumbu[2, 1] else "", labelpad=2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
    simpan(gbr, NAMA)


if __name__ == "__main__":
    main()
