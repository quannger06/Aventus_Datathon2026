# Aventus Datathon 2026

Repository nop bai cho Datathon 2026 Round 1. Repo nay chi chua cac file can
nop: notebook model duoc chon, file code Python duoc convert tu notebook, file
submission va README huong dan chay lai.

## Cau truc file

- `aventus_model_2026.ipynb`: notebook model duoc chon de nop bai.
- `aventus_model_2026.py`: file code Python duoc chuyen tu
  `aventus_model_2026.ipynb`.
- `submission_model.csv`: file submission chinh de nop len Kaggle.
- `README.md`: mo ta cau truc repo va cach chay lai ket qua.

## Mo ta phuong phap

Notebook thuc hien pipeline du bao theo chuoi thoi gian cho cac chi so ban hang,
bao gom:

- doc du lieu dau vao tu `sales.csv` va `sample_submission.csv`;
- tao cac dac trung lich, mua vu, ngay le va cac dac trung tre;
- huan luyen cac model machine learning/time-series trong notebook;
- tao du bao cho giai doan can nop;
- xuat file submission theo dinh dang cua competition.

## Cach chay lai bang notebook

Can cai cac thu vien Python duoc import trong notebook, sau do mo:

```bash
jupyter notebook aventus_model_2026.ipynb
```

Chay lan luot cac cell trong notebook. Notebook goc dang dung path Kaggle:

```python
DIR = "/kaggle/input/competitions/datathon-2026-round-1/"
```

Neu chay local, doi `DIR` thanh duong dan toi thu muc dang chua data, vi du:

```python
DIR = "D:/duong_dan_toi_data/"
```

Thu muc data can co toi thieu:

- `sales.csv`
- `sample_submission.csv`

## Cach chay bang file Python

Co the chay file code da convert:

```bash
python aventus_model_2026.py
```

Neu chay local, sua bien `DIR` o dau file `aventus_model_2026.py` thanh thu muc
data tren may truoc khi chay.

## Submission

File submission chinh trong repo:

```text
submission_model.csv
```

File nay la ket qua duoc dung de nop len Kaggle cho model
`aventus_model_2026`.
