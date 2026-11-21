# Yapay Sinir Ağları Ara Sınavı

UCI Bank Marketing verisinde PyTorch MLP ile vadeli mevduat aboneliği tahmini. Temel seed 31063. Veri bölünmesi, aday üretimi, CV, ağırlık başlangıçları ve bootstrap için kullanılan seed'ler kod ve sonuç dosyalarında kayıtlıdır.

## Google Colab

1. Notebook'ları [01 — EDA](https://colab.research.google.com/github/JustLotus/YapayS-Vize/blob/main/01_eda.ipynb), [02 — Nested CV](https://colab.research.google.com/github/JustLotus/YapayS-Vize/blob/main/02_nested_cv.ipynb) ve [03 — Bağımsız Test](https://colab.research.google.com/github/JustLotus/YapayS-Vize/blob/main/03_final_test.ipynb) sırasıyla açın.
2. Görüntüleme hücreleri teslim edilen sonuçları açar. Yerel çalıştırma için bu depoyu klonlayın, Python 3.12 ortamında `python -m pip install -r requirements.txt` komutunu çalıştırın ve notebook'ları depo kökünden açın.
3. Yeniden eğitim için sonuçların üzerine yazmamak üzere temiz bir kopyada `python src/experiment.py prepare`, `python src/experiment.py outer` ve `python src/experiment.py lock` komutlarını sırasıyla çalıştırın. Veri, UCI kaynağından otomatik indirilir. CPU yeterlidir.

## Bağımsız test

Etiketsiz tahminler ve seçilen yapılandırma GitHub’da ayrı bir commit’te yayımlanır. `python src/experiment.py evaluate` uzak depodaki tahmin ve kaynak dosyalarını yerel kopyayla karşılaştırır. Yeni bir eğitim gerçekleştirildiğinde yeni tahminlerin testten önce yayımlanması gerekir. Test çalışmasından sonra `python src/make_report.py` ile PDF üretilir. Doğrulanan commit ve tahmin özeti `results/evaluation_provenance.json` dosyasındadır.

Veri kaynağı: Moro, Rita ve Cortez (2014), [Bank Marketing, UCI](https://doi.org/10.24432/C5K306). Yalnızca 4.119 satırlık `bank-additional.csv` kullanılır.

Kullanılan CSV resmî UCI arşiviyle birebir doğrulandı. SHA-256 `7e59cf650004d65d1c9d6b08553bad2ee9a9ad70d594f536e3c584ee6ed5df50`.

Deney ortamı Windows 11 (x64), Python 3.12.10 ve AMD Ryzen 7 7800X3D CPU. PyTorch bir CPU iş parçacığıyla çalıştı. Paket sürümleri `requirements.txt` dosyasındadır. Dış kat ve test tablolarındaki süreler ön işleme, eğitim ve tahmin toplamının saniye cinsinden ölçümüdür.
