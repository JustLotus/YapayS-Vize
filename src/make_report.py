"""Render the Turkish technical report directly from experiment artifacts."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
FONTS = [(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", "arial.ttf", "arialbd.ttf"),
         (Path("/usr/share/fonts/truetype/dejavu"), "DejaVuSans.ttf", "DejaVuSans-Bold.ttf")]
fontdir, regular, bold = next(item for item in FONTS if (item[0]/item[1]).is_file())
pdfmetrics.registerFont(TTFont("Text", str(fontdir/regular)))
pdfmetrics.registerFont(TTFont("Bold", str(fontdir/bold)))

styles = getSampleStyleSheet()
for key in ("Normal", "BodyText"):
    styles[key].fontName = "Text"
    styles[key].fontSize = 10
    styles[key].leading = 16
for key in ("Title", "Heading1", "Heading2"):
    styles[key].fontName = "Bold"
    styles[key].textColor = colors.HexColor("#15334d")
styles["Normal"].spaceAfter = 8
styles["Heading1"].spaceAfter = 13
styles["Heading2"].spaceBefore = 11
story = []


def paragraph(text, style="Normal"):
    story.append(Paragraph(escape(str(text)), styles[style]))


def page(title):
    if story:
        story.append(PageBreak())
    paragraph(title, "Heading1")


def read(name):
    return pd.read_csv(ROOT/"results"/f"{name}.csv")


def load(name):
    return json.loads((ROOT/f"{name}.json").read_text(encoding="utf-8"))


def table(title, frame):
    paragraph(title, "Heading2")
    format_value = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    style = styles["Normal"].clone("Cell")
    style.fontSize, style.leading, style.spaceAfter = 8, 10, 0
    style.alignment = TA_CENTER
    rows = [[Paragraph(escape(str(x)), style) for x in frame.columns]]
    rows.extend([Paragraph(escape(format_value(x)), style) for x in row]
                for row in frame.itertuples(index=False, name=None))
    widths = [475 / len(frame.columns)] * len(frame.columns)
    result = Table(rows, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d7e6f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f5f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([result, Spacer(1, 8)])


def image(name, caption, width=430, height=235):
    graphic = Image(str(ROOT/"figures"/name))
    scale = min(width/graphic.imageWidth, height/graphic.imageHeight)
    graphic.drawWidth = scale*graphic.imageWidth
    graphic.drawHeight = scale*graphic.imageHeight
    story.append(graphic)
    paragraph(caption)


def mean_sd(frame, columns):
    return pd.DataFrame({metric: [f"{a:.3f} ± {b:.3f}" for a, b in
                         zip(frame[f"{metric}_mean"], frame[f"{metric}_std"])] for metric in columns})


test = read("test_metrics")
mlp = test.set_index("method").loc["MLP"]
lr = test.set_index("method").loc["LogisticRegression"]
outer = read("outer_cv_results")
outer_summary = read("outer_cv_summary")
final = read("final_inner_search")
seed = read("seed_stability_results")
seed_stats = read("seed_stability_summary")
protocol = load("configs/protocol")
hp = load("results/final_hyperparameters")
prov = load("results/evaluation_provenance")
ci = load("results/bootstrap_intervals")
counts = load("results/eda_counts")
errors = read("error_type_counts").set_index("type")["count"]
configuration = hp["config"]
metrics = ["AP", "ROC_AUC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Balanced_Accuracy"]
labels = dict(ROC_AUC="ROC-AUC", Balanced_Accuracy="Balanced Acc.", Sensitivity="Duyarlılık",
              Specificity="Özgüllük", Accuracy="Accuracy", Precision="Precision", seconds="Süre (sn)")

page("Bank Marketing verisinde MLP ile sınıflandırma")
paragraph("Yapay Sinir Ağları • Yüksek Lisans Ara Sınav Projesi")
paragraph("Özet", "Heading2")
paragraph(f"Bu çalışma vadeli mevduat aboneliğini tahmin etmek için 4.119 gözlem üzerinde PyTorch MLP, lojistik regresyon ve sınıf öncüllerine dayalı bir referansı karşılaştırır. Altı kontrollü ve altı rastgele aday, eşit bütçeli 3×3 iç içe çapraz doğrulamayla incelendi. Development iç CV seçimi {configuration['id']} adayını belirledi. Bağımsız testte birincil MLP için AP {mlp.AP:.3f}, ROC-AUC {mlp.ROC_AUC:.3f} ve Balanced Accuracy {mlp.Balanced_Accuracy:.3f} elde edildi. Lojistik regresyonun AP değeri {lr.AP:.3f} idi.")
paragraph("Araştırma sorusu", "Heading2")
paragraph("Sabit arama bütçesinde ölçekleme, optimizer, ağ genişliği ve öğrenme oranı kontrolü azınlık sınıfının sıralanmasını nasıl etkiler? Bulgular ayrı dış katlarda ve önceden ayrılmış bağımsız testte incelenir.")
paragraph("Giriş", "Heading2")
paragraph("MLP, doğrusal dönüşümler ve doğrusal olmayan aktivasyonları ardışık kullanarak girdi etkileşimlerini temsil eder. Bu esneklik küçük ve dengesiz bir tabloda ek varyansa dönüşebilir. Lojistik regresyon doğrusal bir karşılaştırma noktası, DummyClassifier ise sınıf öncüllerinin sağladığı taban performanstır. Average Precision, pozitif sınıfın sıralamasını precision ve recall üzerinden değerlendirir. Sabit eşikteki Accuracy, negatif sınıf baskın olduğunda yüksek kalabilir.")
paragraph("Deney kaydı", "Heading2")
paragraph(f"Temel seed {protocol['base_seed']}, stratified ayrım seed’i {protocol['split_seed']}, nihai ağırlık seed’leri {', '.join(map(str, protocol['weight_seeds']))}. Birincil seed {hp['primary_seed']}. Bu çalışma yerel CPU üzerinde çalıştırıldı.")
paragraph(f"GitHub tahmin kilidi: {prov['prediction_lock_commit']}. Bu kayıt yeniden yürütülen deneye aittir.")

page("1. Veri seti ve sınıf dağılımı")
paragraph(f"UCI Bank Marketing verisinin 4.119 satırlık bank-additional.csv sürümü kullanıldı. Hedef yes=1, no=0 olarak kodlandı. Görüşme süresi (duration) tahmin anında bilinmediği için çıkarıldı. Stratified ayrımda {counts['development']} development, {counts['test']} bağımsız test satırı belirlendi. Aynı kişi bilgisi verilmediğinden kimlik bazında gruplama yapılamadı.")
table("Tablo 1. Development sınıf dağılımı", read("class_distribution").rename(columns={"class":"Sınıf","count":"Adet","ratio":"Oran"}))
split = read("dataset_split").groupby("split", as_index=False)[["positive_ratio","negative_ratio"]].first()
table("Tablo 2. Ayrım oranları", split.rename(columns={"split":"Bölüm","positive_ratio":"Pozitif","negative_ratio":"Negatif"}))
paragraph("Tablo 1 sınıf dengesizliğini, Tablo 2 stratified ayrımın sınıf oranlarını koruduğunu gösterir. Test oranları yalnızca kilit doğrulandıktan sonra hesaplandı.")
image("eda_class_balance.png", "Şekil 1. Negatif sınıfın sayısal üstünlüğü AP ve sınıf bazlı metriklerle birlikte değerlendirilmelidir.", height=190)
paragraph(f"Development kümesinde hedef ve kullanılan girdiler birlikte incelendiğinde {counts['duplicate_dev_feature_target_rows']} yinelenen satır bulundu. Kaynak indeksleri korunarak rastgele ayrım uygulandı. Tekrarlı ölçüm ve zaman bağımlılığı konusunda bağımsızlık çıkarımı yapılamaz.")

page("2. Keşifsel veri analizi")
numeric = read("numeric_summary")
table("Tablo 3. Sayısal özet ve IQR aykırı değer sayısı",
      numeric[["feature","mean","50%","std","IQR","min","max","IQR_outliers"]].rename(columns={
          "feature":"Değişken","mean":"Ort.","50%":"Medyan","std":"SD","min":"Min","max":"Maks.","IQR_outliers":"Aykırı"}))
largest = numeric.sort_values("IQR_outliers", ascending=False).iloc[0]
paragraph(f"Tablo 3’te en çok işaretlenen değişken {largest.feature} ({int(largest.IQR_outliers)} satır). IQR sınırları Q1 − 1,5 IQR ve Q3 + 1,5 IQR olarak belirlendi. pdays=999 hesaplamaya alınmadı. İşaretlenen satırlar veri setinde tutuldu.")
image("eda_histograms.png", "Şekil 2. Sayısal dağılımlar ölçek ve çarpıklık bakımından farklılaşır.", height=190)
image("eda_boxplots.png", "Şekil 3. Kutu grafikleri uç gözlemlerin yoğunlaştığı değişkenleri görünür kılar.", height=160)

page("3. Özel kodlar ve deney hipotezleri")
special = read("special_values").query("missing > 0 or unknown > 0 or pdays_999 > 0")
table("Tablo 4. Development özel değerleri", special.rename(columns={
    "feature":"Değişken","missing":"Eksik","unknown":"unknown","unique":"Benzersiz","pdays_999":"pdays=999"}))
paragraph("Tablo 4’te unknown, gözlenen ayrı bir kategori olarak tutuldu. pdays=999, önceki temasın bulunmadığını gösteren bir ikili değişkene dönüştürüldü. Sayısal pdays alanında bu kod eksik değer olarak ayrılarak eğitim katının medyanıyla dolduruldu.")
image("eda_correlations.png", "Şekil 4. Birlikte hareket eden makroekonomik değişkenler model katsayıları için yorum sınırı oluşturur.", height=198)
image("eda_categories.png", "Şekil 5. İş kategorilerinin frekansları eşit değildir.", height=118)
paragraph("Hipotezler: StandardScaler, M4’teki MinMax dönüşümünden daha iyi AP sağlayabilir. Momentumlu SGD, ortak öğrenme oranında M1 ve M2’den daha yavaş yakınsayabilir. M5’in geniş ağı ek etkileşimleri yakalayabilir. M6, öğrenme oranı düşürmenin katkısını ölçer.")

page("4. Ön işleme ve MLP yöntemi")
paragraph("Burada normalizasyon, MinMax dönüşümüyle eğitim minimumunu 0’a ve maksimumunu 1’e eşlemeyi ifade eder. Standardizasyon, eğitim ortalamasını çıkarır ve standart sapmaya böler. MinMax uç değerlere doğrudan duyarlıdır. Standardizasyonun ölçeği de uç değerlerle değişebilir. RobustScaler medyan ve IQR kullanır. Yalnızca rastgele adaylarda kırpma arandı ve %1–%99 sınırları ilgili eğitim alt kümesinden öğrenildi.")
paragraph("Sayısal imputasyon kat içi medyanla, kategorik imputasyon en sık kategoriyle yapıldı. One-hot sütunları eğitim alt kümesinde öğrenildi ve görülmeyen kategoriler yok sayıldı. Her MLP eğitiminde ağırlıkları güncelleyen alt küme, BCE izleyen %15 stratified erken durdurma alt kümesinden ayrıldı. Ön işleme yalnız ağırlık eğitimi alt kümesine fit edildi.")
paragraph("Şekil 6. Veri ve model seçimi akışı", "Heading2")
for line in ["Ham veri → sabit stratified development / test ayrımı",
             "Development → 3 dış kat → her dış eğitimde 3 iç kat × 12 aday",
             "İç AP seçimi → dış eğitimde refit → ayrılmış dış kat skorları",
             "Tam development → yeni 3 iç kat → nihai aday ve üç ağırlık seed’i",
             "Etiketsiz test tahminleri → GitHub kilidi → nihai test değerlendirmesi"]:
    paragraph(line)
model = hp["models"][0]
paragraph("Şekil 7. Nihai MLP mimarisi", "Heading2")
paragraph(f"{model['input_width']} girdi → {' → '.join(map(str, configuration['hidden']))} gizli birim ({configuration['activation']}) → 1 logit → sigmoid. Öğrenilebilir parametre: {model['parameters']}. Her kat için girdi × çıktı ağırlıkları ve çıktı başına bir bias sayıldı.")
paragraph("SGD+Momentum geçmiş gradyan yönünü biriktirir. Adam, birinci ve ikinci momentlerle parametre bazlı adım büyüklüğü hesaplar. AdamW ağırlık büzülmesini gradyan güncellemesinden ayırır. Eğitim kaybı BCEWithLogitsLoss’tur. ReduceLROnPlateau iç doğrulama BCE’sini izler, 5 epoch bekler, oranı 0,5 ile çarpar ve 10⁻⁵ tabanını kullanır. Erken durdurma 15 epoch iyileşme olmadığında devreye girer. Üst sınır 150 epoch, karar eşiği 0,50’dir.")

for group, key, number in [("Manual", "manual", 5), ("Random", "random", 7)]:
    page(f"{'5. Kontrollü manuel deneyler' if group == 'Manual' else '6. Tabakalı rastgele arama'}")
    records = []
    for conf in load(f"configs/{key}_candidates"):
        records.append(dict(Aday=conf["id"], Opt=conf["optimizer"], Gizli="/".join(map(str, conf["hidden"])),
                            Scaler=conf["scaler"], LR=f"{conf['lr']:.2g}", WD=f"{conf['wd']:.1g}",
                            Batch=conf["batch"], Akt=conf["activation"], Drop=conf["dropout"],
                            Clip="E" if conf["clip"] else "H", LRdrop="E" if conf["scheduler"] else "H"))
    table(f"Tablo {number}. Sabitlenen adaylar", pd.DataFrame(records))
    summary = final.query("group == @group")[["candidate","mean_AP","sd_AP","seconds","mean_best_epoch"]]
    table(f"Tablo {number+1}. Tam development iç CV sonuçları", summary.rename(columns={
        "candidate":"Aday","mean_AP":"AP ort.","sd_AP":"AP SD","seconds":"Toplam sn","mean_best_epoch":"En iyi epoch ort."}))
    if group == "Manual":
        ap = summary.set_index("candidate").mean_AP
        paragraph(f"Tablo 6’da aynı mekanizma altında M1, M2 ve M3 AP değerleri sırasıyla {ap.M1:.3f}, {ap.M2:.3f}, {ap.M3:.3f}. MinMax etkisi M4−M1 {ap.M4-ap.M1:+.3f}, büyüklük etkisi M5−M1 {ap.M5-ap.M1:+.3f}, scheduler’ı kapatma etkisi M6−M1 {ap.M6-ap.M1:+.3f} AP. Bu farklar iç kat özetleridir.")
    else:
        paragraph("Tablo 7’de her optimizer için tam iki aday bulunur. Öğrenme oranı log ölçeğinde örneklendi. SGD aralığı 10⁻³–10⁻¹, Adam ve AdamW aralığı 10⁻⁴–10⁻². Rastgele adaylar birden fazla bileşende değiştiğinden farklar tek bir değişkene atfedilemez.")
        image("final_candidate_comparison.png", "Şekil 8. Her adayın development iç AP ortalaması ve katlar arası örnek SD.", height=190)
    paragraph(f"Bu grubun en yüksek development iç AP’si {summary.iloc[0].candidate} adayında ({summary.iloc[0].mean_AP:.3f}). Her adayın dış eğitim içindeki kat skorları, süreleri ve epoch bilgisi sonuç CSV’lerinde bulunur.")

page("7. Nested CV bulguları")
raw = outer[["fold","method","candidate","AP","ROC_AUC","F1","Balanced_Accuracy","seconds"]]
table("Tablo 9. Dış kat sonuçları", raw.rename(columns=labels))
main = ["AP","ROC_AUC","F1","Sensitivity","Specificity","Balanced_Accuracy"]
table("Tablo 10. Dış kat ortalaması ± örnek SD",
      pd.concat([outer_summary.method, mean_sd(outer_summary, main)], axis=1).rename(columns=labels))
more = ["Accuracy","Precision","seconds"]
table("Tablo 11. Diğer dış kat metrikleri", pd.concat([outer_summary.method,
      mean_sd(outer_summary, more)], axis=1).rename(columns=labels))
top = outer_summary.sort_values("AP_mean", ascending=False).iloc[0]
paragraph(f"Tablo 10’da en yüksek ortalama dış AP {top.method} için {top.AP_mean:.3f}. Üç katın eğitim kümeleri örtüşür, dolayısıyla SD bir güven aralığı olarak kullanılamaz. Nihai aday dış skor sıralamasıyla değil, tam development üzerinde yapılan yeni iç CV ile seçildi.")
image("outer_ap_comparison.png", "Şekil 9. Dış katlar arasında AP değişimi.", height=136)

page("8. Bağımsız test ve bootstrap")
table("Tablo 12. Bağımsız test metrikleri", test[["method", *metrics]].rename(columns=labels))
table("Tablo 13. Birincil MLP için %95 yüzdelik bootstrap aralıkları", pd.DataFrame([
    {"Metrik": m, "Alt": b[0], "Üst": b[1]} for m,b in ci.items()]))
paragraph(f"Tablo 12’de AP farkı MLP−lojistik regresyon {mlp.AP-lr.AP:+.3f}; duyarlılık {mlp.Sensitivity:.3f}, özgüllük {mlp.Specificity:.3f}. Tablo 13, iki sınıfı içeren 1.000 test yeniden örneklemesinden hesaplandı. Bu aralıklar aynı kilitli tahminlerin örnekleme değişkenliğini yansıtır.")
image("test_confusion_MLP.png", "Şekil 10. Birincil MLP karışıklık matrisi.", height=170)
image("test_confusion_LogisticRegression.png", "Şekil 11. Lojistik regresyon karışıklık matrisi.", height=170)

page("9. Eğriler ve ağırlık başlangıcı kararlılığı")
image("test_roc.png", "Şekil 12. ROC eğrisi duyarlılık ve yanlış pozitif oranını birlikte gösterir.", height=175)
image("test_pr.png", "Şekil 13. PR eğrisi eşik boyunca precision ve recall değişimini gösterir.", height=175)
values = seed[["seed", *metrics]].copy()
values = pd.concat([values, pd.DataFrame([{"seed":"Ort. ± SD", **mean_sd(seed_stats, metrics).iloc[0].to_dict()}])], ignore_index=True)
table("Tablo 14. Üç seed ve örnek standart sapma", values.rename(columns=labels))
paragraph(f"Tablo 14’te başlangıç ağırlığına göre test AP aralığı {seed.AP.min():.3f}–{seed.AP.max():.3f}. Üç model aynı ayrımı ve eşiği kullandı. Ana sonuç önceden belirlenen {hp['primary_seed']} seed’inden alındı. AP, trapez PR-AUC değildir.")

page("10. Yakınsama, hata analizi ve sonuç")
image("final_learning_curves.png", "Şekil 14. Eğitim ve iç doğrulama BCE kayıpları.", height=145)
image("final_learning_rate.png", "Şekil 15. Epoch başında kullanılan öğrenme oranı.", height=145)
paragraph(f"Birincil modelin en iyi epoch’u {model['best_epoch']}, durduğu epoch {model['stopped_epoch']}. Doğrulama kaybı en iyi ağırlıkları belirledi. Sabit eşikte MLP {errors.get('FP',0)} yanlış pozitif ve {errors.get('FN',0)} yanlış negatif üretti. Yanlış negatifler potansiyel abonelerin kaçırılması, yanlış pozitifler gereksiz temas maliyeti doğurabilir. Kayıtlar hata türünü ve tahmin olasılığını örnek düzeyinde gösterir. Gözlemsel veriden nedensel hata açıklaması çıkarılamaz.")
paragraph("Azınlık sınıfı ve küçük test örneği özellikle duyarlılık, precision ve F1 belirsizliğini artırır. Eşik test verisine göre değiştirilmedi. Müşteri kimliği ve kişi bazlı zaman ayrımı bulunmadığından bu performans yeni müşterilere veya farklı kampanya dönemlerine genellenemez.")
paragraph("Kaynaklar ve çalıştırma", "Heading2")
paragraph("Moro, Rita ve Cortez (2014). Bank Marketing. UCI Machine Learning Repository. doi:10.24432/C5K306. PyTorch, ReduceLROnPlateau ve BCEWithLogitsLoss belgeleri. scikit-learn, nested cross-validation, veri sızıntısı ve sınıflandırma metrikleri belgeleri.")
paragraph("Paket sürümleri requirements.txt dosyasında, deney geçmişi ve Colab erişimi README’de kayıtlıdır.")


def footer(canvas, doc):
    canvas.setFont("Text", 8)
    canvas.setFillColor(colors.HexColor("#526579"))
    canvas.drawString(58, 34, "Bank Marketing • Teknik rapor")
    canvas.drawRightString(doc.pagesize[0]-58, 34, str(doc.page))


SimpleDocTemplate(str(ROOT/"report.pdf"), title="Bank Marketing verisinde MLP ile sınıflandırma",
                  leftMargin=54, rightMargin=54, topMargin=48, bottomMargin=55).build(
                      story, onFirstPage=footer, onLaterPages=footer)
