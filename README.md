# Aventus Datathon 2026

Repository nop bai cho Datathon 2026 Round 1. Repo nay chua notebook model,
file Python duoc convert tu notebook, notebook visualization/EDA va file
submission.

## Cau truc file

- `aventus_model_2026.ipynb`: notebook model duoc chon de tao submission.
- `aventus_model_2026.py`: file code Python duoc chuyen tu
  `aventus_model_2026.ipynb`.
- `Visualization_Aventus_Datathon.ipynb`: notebook EDA/visualization, tao cac
  bieu do phan tich kinh doanh va du phong.
- `submission_model.csv`: file submission chinh.
- `README.md`: mo ta cau truc repo va cach chay lai.

## Model

Model trong `aventus_model_2026.ipynb` su dung pipeline du bao chuoi thoi gian:

- doc du lieu tu `sales.csv` va `sample_submission.csv`;
- tao feature lich, mua vu, sale season va peak season;
- tach trend bang Prophet;
- hoc residual bang XGBoost, LightGBM va CatBoost;
- stack ket qua bang HuberRegressor;
- xuat submission theo format competition.

File Python `aventus_model_2026.py` la ban convert tu notebook model, phu hop
khi can xem code duoi dang script.

## Visualization

Notebook `Visualization_Aventus_Datathon.ipynb` duoc viet theo workflow Colab.
Notebook nay doc 13 bang raw:

- `products.csv`
- `customers.csv`
- `orders.csv`
- `order_items.csv`
- `payments.csv`
- `shipments.csv`
- `returns.csv`
- `reviews.csv`
- `sales.csv`
- `inventory.csv`
- `web_traffic.csv`
- `promotions.csv`
- `geography.csv`

Mac dinh notebook dung cac path:

```python
DATASET_PATH = "/content/drive/MyDrive/Datathon_2026/Dataset"
OUTPUT_PATH  = "/content/drive/MyDrive/Datathon_2026/EDA_Final"
```

Neu chay local hoac dung Drive folder khac, can doi 2 bien tren cho dung voi
vi tri data va folder output.

Notebook visualization tao cac chart:

- `M1_master_combo.png`: dien bien doanh thu va traffic website 2012-2022.
- `M3_What_If_CR.png`: what-if scenario khi phuc hoi conversion rate.
- `M4_Promo_Mismatch.png`: lech pha khuyen mai theo thang/mua vu.
- `M6_ABC_XYZ_Matrix.png`: ma tran ABC-XYZ cho ton kho.
- `M8_Impact_Effort.png`: ma tran uu tien thuc thi.
- `M9_Projection.png`: du phong tang truong 2023-2024.
- `D1_Purchase_Frequency_Drop.png`: diagnostic tan suat mua hang.
- `D2_Promo_AOV_Trap.png`: diagnostic AOV khi dung promotion.
- `D3_Sold_vs_Stock_Category.png`: diagnostic sold vs stock theo nganh hang.

## Cach chay model bang notebook

Mo notebook:

```bash
jupyter notebook aventus_model_2026.ipynb
```

Notebook goc dang dung path Kaggle:

```python
DIR = "/kaggle/input/competitions/datathon-2026-round-1/"
```

Neu chay local, doi `DIR` thanh duong dan toi folder data:

```python
DIR = "D:/duong_dan_toi_data/"
```

Folder data can co toi thieu:

- `sales.csv`
- `sample_submission.csv`

## Cach chay model bang Python

Co the chay file script:

```bash
python aventus_model_2026.py
```

Truoc khi chay local, sua bien `DIR` o dau file
`aventus_model_2026.py` thanh folder data tren may.

## Cach chay visualization

Mo notebook:

```bash
jupyter notebook Visualization_Aventus_Datathon.ipynb
```

Hoac upload notebook len Google Colab. Neu dung Colab, mount Google Drive va
dam bao `DATASET_PATH` tro toi folder chua 13 file CSV raw. Cac chart PNG se
duoc luu vao `OUTPUT_PATH`.

## Submission

File submission chinh:

```text
submission_model.csv
```

File nay la output cua model `aventus_model_2026`. Neu ban to chuc chi cham
`Revenue`, cot `Revenue` trong file submission la cot can quan tam. Cot `COGS`
duoc giu theo format `sample_submission.csv` hien co.

## Luu y

Data raw khong duoc dua vao repo nay. De chay lai day du, can dat data theo
dung path trong notebook/script hoac cap nhat cac bien path truoc khi chay.
