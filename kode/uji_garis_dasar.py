import numpy as np
from sklearn.dummy import DummyClassifier

from data import foto_berlabel, tulis_json
from model import prediksi_loo, selang_wilson

KELUARAN = "uji_garis_dasar.json"


def kelas_mayoritas(y):
    kelas, jumlah = np.unique(y, return_counts=True)
    benar = int(jumlah.max())
    tebakan_loo = prediksi_loo(DummyClassifier(strategy="most_frequent"), np.zeros((len(y), 1)), y)
    return {"n": len(y), "n_per_kelas": dict(zip(kelas.tolist(), jumlah.tolist())),
            "kelas_mayoritas": kelas[jumlah == benar].tolist(), "benar": benar, "akurasi": benar / len(y),
            "wilson95": selang_wilson(benar, len(y)), "peluang_acak": 1 / len(kelas),
            "loo_dummy_most_frequent_benar": int(np.sum(tebakan_loo == y))}


def main():
    foto = foto_berlabel()
    y = np.array([p["kelas"] for _, p in foto])
    f15 = np.array([p["lensa"] == "f15" for _, p in foto])
    tulis_json(KELUARAN, {"semua": kelas_mayoritas(y), "f15_saja": kelas_mayoritas(y[f15])})


if __name__ == "__main__":
    main()
