import re
from collections import Counter, defaultdict
from itertools import combinations

import cv2
import numpy as np

from data import BATAS_V_SPEKTRUM, baca_bgr, exif_ringkas, jalur_relatif, jejak_jpeg, peran_citra, rerata_cara_ta, semua_citra, sha256, tulis_json

KELUARAN = "uji_inventaris.json"
UKURAN_SIDIK = (48, 64)
BATAS_MIRIP = 0.95
TOLERANSI_RASIO = 0.01
POLA_TANGGAL_NAMA = r"(\d{4}-\d{2}-\d{2}) at (\d{2})\.(\d{2})\.(\d{2})"


def sidik_abu(bgr):
    abu = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kecil = cv2.resize(abu, UKURAN_SIDIK, interpolation=cv2.INTER_AREA).astype(float).ravel()
    return (kecil - kecil.mean()) / kecil.std()


def catat_citra(jalur):
    bgr = baca_bgr(jalur)
    peran = peran_citra(jalur)
    rerata = rerata_cara_ta(bgr)
    if peran["peran"] == "foto f.20 tanpa label":
        peran["peran"] = "foto spektrum tanpa label" if rerata["V"] < BATAS_V_SPEKTRUM else "foto susunan alat"
    tinggi, lebar = bgr.shape[:2]
    tanggal_nama = re.search(POLA_TANGGAL_NAMA, jalur.name)
    return {"jalur": jalur_relatif(jalur), **peran, "lebar": lebar, "tinggi": tinggi, "sha256": sha256(jalur),
            "exif": exif_ringkas(jalur), "jpeg": jejak_jpeg(jalur) if jalur.suffix.lower() == ".jpg" else None,
            "tanggal_dari_nama": f"{tanggal_nama[1]} {tanggal_nama[2]}:{tanggal_nama[3]}:{tanggal_nama[4]}" if tanggal_nama else None,
            "rerata_V_seluruh": rerata["V"]}, sidik_abu(bgr)


def kelompok_hash(catatan):
    per_hash = defaultdict(list)
    for isi in catatan:
        per_hash[isi["sha256"]].append(isi["jalur"])
    return [jalur for jalur in per_hash.values() if len(jalur) > 1]


def pasangan_mirip(catatan, sidik):
    foto = [i for i, isi in enumerate(catatan) if isi["peran"].startswith("foto")]
    pasangan = []
    for a, b in combinations(foto, 2):
        rasio_a = catatan[a]["lebar"] / catatan[a]["tinggi"]
        rasio_b = catatan[b]["lebar"] / catatan[b]["tinggi"]
        if abs(rasio_a - rasio_b) > TOLERANSI_RASIO * rasio_a:
            continue
        r = float(np.mean(sidik[a] * sidik[b]))
        if r >= BATAS_MIRIP:
            pasangan.append({"a": catatan[a]["jalur"], "b": catatan[b]["jalur"], "r": r})
    return pasangan


def jumlah_adegan(jalur_kelas, pasangan):
    induk = {j: j for j in jalur_kelas}

    def akar(j):
        while induk[j] != j:
            j = induk[j]
        return j

    for p in pasangan:
        if p["a"] in induk and p["b"] in induk:
            induk[akar(p["a"])] = akar(p["b"])
    return len({akar(j) for j in jalur_kelas})


def rekap_sampel(catatan, pasangan):
    berlabel = [isi for isi in catatan if isi["peran"] == "foto spektrum"]
    rekap = {}
    for kelas in sorted({isi["kelas"] for isi in berlabel}):
        rekap[kelas] = {}
        for lensa in ("f15", "f20"):
            pilih = [isi for isi in berlabel if isi["kelas"] == kelas and isi["lensa"] == lensa]
            rekap[kelas][lensa] = {"citra": len(pilih), "hash_unik": len({isi["sha256"] for isi in pilih}),
                                   "adegan_setelah_gabung_mirip": jumlah_adegan([isi["jalur"] for isi in pilih], pasangan)}
    return rekap


def rekap_jejak(catatan):
    rekap = defaultdict(lambda: {"jumlah": 0, "sisi_panjang": set(), "progresif": set(), "exif_model": set(), "peran": set()})
    for isi in catatan:
        if not isi["jpeg"]:
            continue
        r = rekap[isi["jpeg"]["tabel_kuantisasi"]]
        r["jumlah"] += 1
        r["sisi_panjang"].add(max(isi["lebar"], isi["tinggi"]))
        r["progresif"].add(isi["jpeg"]["progresif"])
        r["peran"].add(isi["peran"])
        if isi["exif"]["model"]:
            r["exif_model"].add(f'{isi["exif"]["merek"]} {isi["exif"]["model"]}')
    return {k: {a: sorted(b) if isinstance(b, set) else b for a, b in v.items()} for k, v in rekap.items()}


def main():
    hasil = [catat_citra(jalur) for jalur in semua_citra()]
    catatan = [h[0] for h in hasil]
    sidik = [h[1] for h in hasil]
    pasangan = pasangan_mirip(catatan, sidik)
    keluaran = {"batas_V_spektrum": BATAS_V_SPEKTRUM, "batas_korelasi_mirip": BATAS_MIRIP, "ukuran_sidik": list(UKURAN_SIDIK),
                "jumlah_berkas_citra": len(catatan), "per_peran": dict(Counter(isi["peran"] for isi in catatan)),
                "exif_berisi_tanggal": sum(1 for isi in catatan if isi["exif"]["tanggal"] or isi["exif"]["tanggal_asli"]),
                "exif_berisi_model": dict(Counter(f'{isi["exif"]["merek"]} {isi["exif"]["model"]}' for isi in catatan if isi["exif"]["model"])),
                "duplikat_byte": kelompok_hash(catatan), "jejak_kuantisasi": rekap_jejak(catatan), "pasangan_hampir_sama": pasangan,
                "sampel_per_kelas_lensa": rekap_sampel(catatan, pasangan), "citra": catatan}
    tulis_json(KELUARAN, keluaran)


if __name__ == "__main__":
    main()
