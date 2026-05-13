# 🌧️ LiDAR Nokta Bulutlarının Olumsuz Hava Koşullarına Karşı Dayanıklılığı

> **Ders:** Deep Learning — YZM 306 (2)  
> **Üniversite:** Ostim Teknik Üniversitesi — Yapay Zeka Mühendisliği  
> **Öğrenci:** Osman ACAR  
> **Model:** PointPillars (sıfırdan) | **Veri Seti:** KITTI 3D Object Detection

---

## 📌 Proje Özeti

Özerk sürüş sistemleri 3D algılama için büyük ölçüde LiDAR sensörlerine dayanır. Ancak PointPillars, VoxelNet ve CenterPoint gibi modern 3D nesne tespiti modelleri, yalnızca **temiz hava veri setleri** (KITTI, nuScenes) üzerinde eğitilip değerlendirilmektedir. Bu durum kritik bir güvenlik açığı yaratmaktadır: gerçek dünya koşullarında sis, yağmur ve kar, LiDAR dönüş özelliklerini sinyal zayıflaması, geri saçılım ve sahte tespitler yoluyla köklü biçimde değiştirmektedir.

Bu çalışma şu araştırma sorularını ele almaktadır:

1. **Ne kadar ağır?** — Sis, yağmur ve kar, PointPillars'ın iç özellik temsillerini ne ölçüde bozar?
2. **Kırılma noktası nerede?** — Modelin performansı hangi şiddette çöker?
3. **Hava koşulları artırmalı eğitim** dayanıklılığı artırıyor mu, temiz hava performansından ne kadar taviz veriyor?
4. **DROR ön filtrelemesi** simülasyon bozukluklarından kayıp performansı geri kazandırabilir mi?

---

## 🗂️ Proje Yapısı

```
├── LiDAR_Adverse_Weather_Robustness_FINAL.ipynb   # Ana notebook (tüm deneyler)
├── README.md
└── results/                                        # Üretilen grafikler (Drive'dan)
    ├── robustness_results.json
    ├── per_class_robustness.png
    └── tipping_point_all_weather.png
```

---

## 🔬 Yöntem & Pipeline

```
KITTI Temiz Veri ──► PointPillars (Baseline) ──► Özellik Benzerliği @ Temiz
       │                                                      │
       ▼                                                      ▼
Hava Simülasyonu ──► Sis / Yağmur / Kar ──► Özellik Benzerliği @ Bozuk
(3 tür × 3 seviye)        │                               │
(Beer-Lambert, Kunkel)     ▼                               ▼
                   Artırmalı İnce Ayar ──► Özellik Benzerliği @ İyileştirilmiş
                              │
                              ▼
                   DROR Ön Filtreleme Analizi ──► Domain Gap Bulguları
```

### Hava Simülasyon Modelleri

| Hava Koşulu | Fizik Modeli | Referans |
|-------------|-------------|----------|
| **Sis** | Beer-Lambert zayıflama yasası: `P = P₀ · exp(-2βd)` | Hahner et al. (ICCV 2021) |
| **Yağmur** | Kunkel (1984) ampirik modeli: `β = 0.01 × R^0.6` | Kunkel (Applied Optics 1984) |
| **Kar** | Genişletilmiş saçılım + zemin birikim modeli | Hahner et al. (2022) |

### Şiddet Seviyeleri

| Hava | Hafif | Orta | Ağır |
|------|-------|------|------|
| Sis (β) | 0.02 (~200m görüş) | 0.06 (~70m görüş) | 0.15 (~25m görüş) |
| Yağmur (mm/sa) | 2 | 25 | 100 |
| Kar (mm/sa su-eş) | 0.5 | 2.0 | 8.0 |

---

## 📊 Temel Bulgular

### 1. Özellik Haritası Benzerliği (Küresel)
Ağır koşullarda BEV özellik haritası cosinüs benzerliği **0.761–0.981** aralığına düşmekte; en büyük bozulma ağır sis altında gözlemlenmektedir.

### 2. Sınıf Bazında Yerel Dayanıklılık
Nesne düzeyinde yerel benzerlik değerleri (0.990–1.000), küresel benzerlik değerlerinden önemli ölçüde yüksek kalmaktadır. Bu bulgu önemli bir mimari özelliği ortaya koymaktadır:

> **Hava bozulması, özellik haritasında küresel mekânsal yeniden düzenlemeye yol açarken yerel nesne düzeyindeki temsiller görece stabil kalmaktadır.**

### 3. Kırılma Noktası Analizi

| Hava | Metrik | Baseline Kırılma | Artırmalı Kırılma | Δ |
|------|--------|-----------------|-------------------|---|
| **Sis** | β katsayısı | 0.060 | **0.110** | **+%83** |
| **Kar** | mm/sa (su-eş) | 6.0 | 6.0 | 0% |
| **Yağmur** | mm/sa | 25.0 | 25.0 | 0% |

**Kritik Bulgu:** Hava artırmalı eğitim yalnızca sis için kırılma noktasını öteler; kar ve yağmur için etki yoktur.

### 4. DROR Analizi
DROR ön filtrelemesi, simülasyon verisi üzerinde sınırlı iyileştirme sağlamıştır. Bunun temel nedeni: simülasyonumuz yalnızca **zayıflama tabanlı bozulmayı** (nokta kaldırma) modellemekte, gerçek dünyadaki **geri saçılım gürültüsünü** (sahte nokta ekleme) içermemektedir. Bu asimetri, DROR'un etkinliğini doğrudan sınırlamaktadır.

---

## ⚙️ Kurulum & Çalıştırma

### Gereksinimler

```bash
# Python ortamı
pip install open3d spconv-cu118 pyquaternion
pip install scikit-learn scikit-image matplotlib seaborn plotly tqdm

# OpenPCDet (PointPillars altyapısı)
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && pip install -r requirements.txt && python setup.py develop
```

### KITTI Veri Seti
1. [KITTI 3D Object Detection](http://www.cvlibs.net/datasets/kitti/) adresinden lisans kabul ederek indirin.
2. Velodyne, label_2, calib ve image_2 dosyalarını `data/kitti/` altına çıkartın.

### Notebook Çalıştırma

```bash
# Google Colab önerilir (A100/H100 GPU)
# Runtime → Change runtime type → A100
jupyter notebook LiDAR_Adverse_Weather_Robustness_FINAL.ipynb
```

**Bölüm sırası:**
- **Section 0** — Ortam kurulumu & GPU doğrulama
- **Section 1** — KITTI veri yükleyici
- **Section 2** — Hava simülasyon motoru
- **Section 7** — Özellik benzerliği analizi (baseline)
- **Section 11** — DROR filtreleme analizi
- **Section 12** — Sis kırılma noktası taraması
- **Section 14** — Artırmalı eğitim karşılaştırması
- **Section 15** — Sınıf bazında dayanıklılık
- **Section 16** — Çok hava tipli kırılma noktası analizi

---

## 📚 Referanslar

1. Lang et al. (2019). *PointPillars: Fast Encoders for Object Detection from Point Clouds.* CVPR.
2. Zhou & Tuzel (2018). *VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection.* CVPR.
3. Geiger et al. (2012). *Are we ready for autonomous driving? The KITTI vision benchmark suite.* CVPR.
4. Hahner et al. (2021). *Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection.* ICCV.
5. Kunkel (1984). *Attenuation of Light by Water Drops.* Applied Optics.
6. Bijelic et al. (2020). *Seeing Through Fog Without Seeing Fog.* CVPR.
7. Charron et al. (2018). *De-noising of LiDAR Point Clouds Corrupted by Snow.* CRV.

---

## 📄 Lisans

Bu proje YZM 306 Derin Öğrenme dersi kapsamında eğitim amaçlı hazırlanmıştır.
