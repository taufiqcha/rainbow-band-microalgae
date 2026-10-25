import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

AKAR = Path(__file__).resolve().parent.parent
TA = AKAR / "data"
HASIL = AKAR / "results"
GAMBAR = HASIL / "gambar"
SEED = 42
KANAL = ("R", "G", "B", "H", "S", "V")
SPESIES = {"Chorella": "Chlorella", "Chlorella": "Chlorella", "Nanno": "Nannochloropsis",
           "Nannochloropsis": "Nannochloropsis", "Spirullina": "Spirulina", "Spirulina": "Spirulina"}
AWALAN_POTONGAN = {"c": "Chlorella", "n": "Nannochloropsis", "s": "Spirulina"}
BATAS_V_SPEKTRUM = 50.0
EKSTENSI_CITRA = (".jpg", ".jpeg", ".png")
LABEL_KONFIRMASI = {"WhatsApp Image 2024-06-19 at 02.44.10_df514fb0": "Spirulina",
                         "WhatsApp Image 2024-06-19 at 02.44.11_d78d8998": "Spirulina"}
SUMBER_LABEL = {"folder": "nama folder F.15", "awalan": "awalan nama berkas f.20",
                "konfirmasi": "konfirmasi penulis TA"}


def jalur_relatif(jalur):
    return str(Path(jalur).relative_to(AKAR))


def semua_citra():
    return sorted(p for p in TA.rglob("*") if p.suffix.lower() in EKSTENSI_CITRA)


def baca_bgr(jalur):
    return cv2.imdecode(np.fromfile(str(jalur), dtype=np.uint8), cv2.IMREAD_COLOR)


def rerata_cara_ta(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {"R": float(np.mean(bgr[:, :, 2])), "G": float(np.mean(bgr[:, :, 1])), "B": float(np.mean(bgr[:, :, 0])),
            "H": float(np.mean(hsv[:, :, 0])), "S": float(np.mean(hsv[:, :, 1])), "V": float(np.mean(hsv[:, :, 2]))}


def sha256(jalur):
    return hashlib.sha256(Path(jalur).read_bytes()).hexdigest()


def exif_ringkas(jalur):
    with Image.open(jalur) as citra:
        exif = citra.getexif()
        rinci = exif.get_ifd(0x8769)
        return {"merek": exif.get(271), "model": exif.get(272), "tanggal": exif.get(306),
                "tanggal_asli": rinci.get(36867), "orientasi": exif.get(274), "perangkat_lunak": exif.get(305)}


def jejak_jpeg(jalur):
    with Image.open(jalur) as citra:
        tabel = getattr(citra, "quantization", None)
        sidik = hashlib.sha256(repr(sorted((k, tuple(v)) for k, v in tabel.items())).encode()).hexdigest()[:8] if tabel else None
        return {"tabel_kuantisasi": sidik, "progresif": bool(citra.info.get("progressive")),
                "penanda_app": [nama for nama, _ in getattr(citra, "applist", [])]}


def peran_citra(jalur):
    bagian = Path(jalur).relative_to(TA).parts
    nama = Path(jalur).stem
    if jalur.suffix.lower() == ".png":
        return {"peran": "grafik keluaran TA", "kelas": None, "lensa": None}
    if bagian[0] == "Testing" and len(bagian) == 4 and bagian[1] == "F.15":
        return {"peran": "foto spektrum", "kelas": SPESIES[bagian[2]], "lensa": "f15", "sumber_label": SUMBER_LABEL["folder"]}
    if bagian[0] == "Testing" and bagian[1] == "f.20":
        if nama in LABEL_KONFIRMASI:
            return {"peran": "foto spektrum", "kelas": LABEL_KONFIRMASI[nama], "lensa": "f20", "sumber_label": SUMBER_LABEL["konfirmasi"]}
        if nama.startswith("WhatsApp"):
            return {"peran": "foto f.20 tanpa label", "kelas": None, "lensa": "f20"}
        return {"peran": "foto spektrum", "kelas": AWALAN_POTONGAN[nama[0]], "lensa": "f20", "sumber_label": SUMBER_LABEL["awalan"]}
    if bagian[-2] in ("Testing", "Images"):
        return {"peran": "potongan ROI", "kelas": AWALAN_POTONGAN.get(nama[0]), "lensa": None}
    if bagian[-2] == "Hasil":
        return {"peran": "potongan uji air (Bab 4.1.1.2)", "kelas": None, "lensa": None}
    return {"peran": "lain", "kelas": None, "lensa": None}


def foto_berlabel():
    return [(jalur, peran) for jalur in semua_citra() if (peran := peran_citra(jalur))["peran"] == "foto spektrum"]






def baca_json(nama):
    return json.loads((HASIL / nama).read_text(encoding="utf-8"))


def tulis_json(nama, isi):
    HASIL.mkdir(exist_ok=True)
    (HASIL / nama).write_text(json.dumps(isi, indent=1, ensure_ascii=False), encoding="utf-8")


def simpan_gambar(gbr, nama):
    GAMBAR.mkdir(parents=True, exist_ok=True)
    gbr.savefig(GAMBAR / f"{nama}.pdf", metadata={"CreationDate": None, "ModDate": None})
    gbr.savefig(GAMBAR / f"{nama}.png", dpi=200)
    return f"results/gambar/{nama}.pdf"
