# Converted from aventus_model_2026.ipynb
# Source notebook: Model/aventus_model_2026.ipynb

# %% cell 1
DIR = "/kaggle/input/competitions/datathon-2026-round-1/"

# %% cell 2
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [markdown] cell 3
# # 1. Helper functions

# %% cell 4
SALE_SEASONS = [
    {'month': 1,  'start_day': 30, 'duration': 30, 'profit_rank': 1},
    {'month': 3,  'start_day': 18, 'duration': 30, 'profit_rank': 2},
    {'month': 6,  'start_day': 23, 'duration': 29, 'profit_rank': 3},
    {'month': 7,  'start_day': 30, 'duration': 34, 'profit_rank': 5},
    {'month': 8,  'start_day': 30, 'duration': 32, 'profit_rank': 4},
    {'month': 11, 'start_day': 18, 'duration': 45, 'profit_rank': 6},
]

# %% cell 5
def add_seasonality_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    
    # ── Basic calendar ──────────────────────────────────────────────────────
    df['year']      = df['Date'].dt.year
    df['month']     = df['Date'].dt.month
    df['day']       = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    
    # ── is_peak_season ────────────────────────────────────────────────────────
    # Tháng 4, 5, 11 là peak season theo EDA (stockout + order surge)
    df['is_peak_season'] = df['Date'].dt.month.isin([4, 5, 11]).astype(int)
    
    # ── peak_proximity ────────────────────────────────────────────────────────
    # Khoảng cách (ngày) đến peak season gần nhất, decay dạng inverse
    # Peak months: tháng 4 (ngày 1/4), tháng 5 (1/5), tháng 11 (1/11)
    def days_to_nearest_peak(date):
        year = date.year
        candidates = [
            pd.Timestamp(year,     4,  1),
            pd.Timestamp(year,     5,  1),
            pd.Timestamp(year,    11,  1),
            pd.Timestamp(year - 1, 4,  1),
            pd.Timestamp(year - 1, 5,  1),
            pd.Timestamp(year - 1, 11, 1),
            pd.Timestamp(year + 1, 4,  1),
            pd.Timestamp(year + 1, 5,  1),
            pd.Timestamp(year + 1, 11, 1),
        ]
        return min(abs((date - c).days) for c in candidates)
    
    days_to_peak = df['Date'].map(days_to_nearest_peak)
    df['peak_proximity'] = 1 / (1 + days_to_peak)
    
    # ── month_sin / month_cos ─────────────────────────────────────────────────
    # Fourier encoding tháng: tránh 12 dummy variables, preserve circular nature
    month = df['Date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    return df

# %% cell 6
def _get_sale_windows(years):
    """
    Sinh ra list các (start_date, end_date, profit_rank) cho từng đợt sale,
    theo từng năm trong `years`.
    Xử lý trường hợp end_date tràn sang tháng sau (ví dụ tháng 7 + 34 ngày).
    """
    windows = []
    for year in years:
        for s in SALE_SEASONS:
            try:
                start = pd.Timestamp(year=year, month=s['month'], day=s['start_day'])
            except ValueError:
                # start_day vượt quá số ngày trong tháng → dùng ngày cuối tháng
                start = pd.Timestamp(year=year, month=s['month'], day=1) + pd.offsets.MonthEnd(0)
            end = start + pd.Timedelta(days=s['duration'] - 1)
            windows.append((start, end, s['profit_rank']))
    return windows

# %% cell 7
def add_sale_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # ── Sale season windows ─────────────────────────────────────────────────
    years   = df['Date'].dt.year.unique()
    windows = _get_sale_windows(years)

    # Khởi tạo
    df['is_sale_season']      = 0
    df['sale_rank']           = 0      # 0 = không sale
    df['sale_progress']       = 0.0   # 0→1 trong đợt sale
    df['days_to_next_sale']   = 999
    df['days_since_last_sale']= 999

    dates = df['Date'].values  # numpy array để vectorise

    for start, end, rank in windows:
        mask_in = (df['Date'] >= start) & (df['Date'] <= end)

        # is_sale_season & sale_rank
        df.loc[mask_in, 'is_sale_season'] = 1
        df.loc[mask_in, 'sale_rank']      = rank

        # days_to_next_sale: cập nhật cho các ngày TRƯỚC đợt sale này
        mask_before = df['Date'] < start
        days_to = (start - df.loc[mask_before, 'Date']).dt.days
        df.loc[mask_before, 'days_to_next_sale'] = np.minimum(
            df.loc[mask_before, 'days_to_next_sale'], days_to
        )

        # days_since_last_sale: cập nhật cho các ngày SAU đợt sale này
        mask_after = df['Date'] > end
        days_since = (df.loc[mask_after, 'Date'] - end).dt.days
        df.loc[mask_after, 'days_since_last_sale'] = np.minimum(
            df.loc[mask_after, 'days_since_last_sale'], days_since
        )

    # Với ngày đang trong sale: days_to = 0, days_since = 0
    df.loc[df['is_sale_season'] == 1, 'days_to_next_sale']    = 0
    df.loc[df['is_sale_season'] == 1, 'days_since_last_sale'] = 0

    return df

# %% cell 8
def add_features(df):
    df = df.copy()
    df = add_seasonality_features(df)
    df = add_sale_features(df)

    return df

# %% [markdown] cell 9
# # 2. Tune base models

# %% cell 10
def tune_base_models(train_path):
    print("1. Chuẩn bị dữ liệu và Pre-calculate Folds...")
    df = pd.read_csv(train_path)
    df['Revenue_log'] = np.log1p(df['Revenue'])
    df['COGS_Ratio'] = df['COGS'] / (df['Revenue'] + 1e-9)
    df = add_features(df)
    
    static_features = [
        'year', 'month', 'day', 'dayofweek',
        "month_sin", "month_cos", 'is_peak_season', 'peak_proximity', 
        'is_sale_season', 'sale_rank', 'days_to_next_sale' , 'days_since_last_sale'
    ]
    
    folds = [
        ('2012-07-04', '2018-12-31', '2019-01-01', '2019-12-31'),
        ('2012-07-04', '2019-12-31', '2020-01-01', '2020-12-31'),
        ('2012-07-04', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2012-07-04', '2021-12-31', '2022-01-01', '2022-12-31')
    ]
    
    cached_folds = []
    for train_start, train_end, val_start, val_end in folds:
        train_fold = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)].copy()
        val_fold = df[(df['Date'] >= val_start) & (df['Date'] <= val_end)].copy()
        
        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        prophet_df = train_fold[['Date', 'Revenue_log']].rename(columns={'Date': 'ds', 'Revenue_log': 'y'})
        prophet.fit(prophet_df)
        
        train_fold['residual'] = train_fold['Revenue_log'] - prophet.predict(prophet_df)['yhat'].values
        val_trend = prophet.predict(val_fold[['Date']].rename(columns={'Date': 'ds'}))['yhat'].values
        val_fold['residual'] = val_fold['Revenue_log'] - val_trend
        
        features = static_features 
        cached_folds.append({
            'X_train': train_fold[features], 'y_train_res': train_fold['residual'],
            'X_val': val_fold[features], 'y_val_res': val_fold['residual'],
            'val_trend': val_trend, 'y_val_true_rev': val_fold['Revenue']
        })

    # ==========================================
    # TUNE XGBOOST (HUBER LOSS)
    # ==========================================
    def objective_xgb(trial):
        params = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'reg:pseudohubererror', 'device': 'cuda', 'tree_method': 'hist', 'random_state': 42
        }
        cv_maes = []
        for fold in cached_folds:
            model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
            model.fit(fold['X_train'], fold['y_train_res'], eval_set=[(fold['X_val'], fold['y_val_res'])], verbose=False)
            val_pred_rev = np.expm1(fold['val_trend'] + model.predict(fold['X_val'])) 
            cv_maes.append(mean_absolute_error(fold['y_val_true_rev'], val_pred_rev))
        return np.mean(cv_maes)

    # ==========================================
    # TUNE LIGHTGBM (HUBER LOSS)
    # ==========================================
    def objective_lgb(trial):
        params = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'huber', 'device': 'gpu', 'random_state': 42, 'verbose': -1
        }
        cv_maes = []
        for fold in cached_folds:
            model = lgb.LGBMRegressor(**params)
            model.fit(fold['X_train'], fold['y_train_res'], eval_set=[(fold['X_val'], fold['y_val_res'])], callbacks=[lgb.early_stopping(50, verbose=False)])
            val_pred_rev = np.expm1(fold['val_trend'] + model.predict(fold['X_val']))
            cv_maes.append(mean_absolute_error(fold['y_val_true_rev'], val_pred_rev))
        return np.mean(cv_maes)

    # ==========================================
    # TUNE CATBOOST (HUBER LOSS)
    # ==========================================
    def objective_cat(trial):
        params = {
            'iterations': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'loss_function': 'Huber:delta=1.5', 'task_type': 'GPU', 'random_state': 42, 'verbose': False
        }
        cv_maes = []
        for fold in cached_folds:
            model = CatBoostRegressor(**params, early_stopping_rounds=50)
            model.fit(fold['X_train'], fold['y_train_res'], eval_set=[(fold['X_val'], fold['y_val_res'])], verbose=False)
            val_pred_rev = np.expm1(fold['val_trend'] + model.predict(fold['X_val']))
            cv_maes.append(mean_absolute_error(fold['y_val_true_rev'], val_pred_rev))
        return np.mean(cv_maes)
    print("Đang Tune XGBoost...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=30)
    
    print("\nĐang Tune LightGBM...")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=30)
    
    print("\nĐang Tune CatBoost...")
    study_cat = optuna.create_study(direction="minimize")
    study_cat.optimize(objective_cat, n_trials=30)

    print("\nXong.")
    print("\n[KẾT QUẢ REVENUE PARAMS ĐỂ COPY VÀO STACKING]")
    print(f"xgb_params_rev = {study_xgb.best_trial.params}")
    print(f"lgb_params_rev = {study_lgb.best_trial.params}")
    print(f"cat_params_rev = {study_cat.best_trial.params}")
    return study_xgb.best_trial.params, study_lgb.best_trial.params, study_cat.best_trial.params

# %% cell 11
# best_xgb_params, best_lgb_params, best_cat_params = tune_base_models(DIR + "sales.csv")

# %% cell 12
best_xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.07687359172082428,
    "max_depth": 4,
    "objective": "reg:pseudohubererror",
    #"device": "cuda",
    "tree_method": "hist",
    "random_state": 42,
    "subsample": 0.9640163894074474,
    "colsample_bytree": 0.9988728404218791,
    "min_child_weight": 6
}

best_lgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.010074446630670307,
    "max_depth": 4,
    "objective": "huber",
    #"device": "gpu",
    "random_state": 42,
    "verbose": -1,
    "num_leaves": 28,
    "subsample": 0.932752195699773,
    "colsample_bytree": 0.8528851999663052
}

best_cat_params = {
    "iterations": 2000,
    "learning_rate": 0.09865972508027333,
    "depth": 4,
    "loss_function": "Huber:delta=1.5",
    #"task_type": "GPU",
    "random_state": 42,
    "verbose": False,
    "l2_leaf_reg": 5.35578476395823
}

# %% [markdown] cell 13
# # 3. Train meta model and predict

# %% cell 14
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV

# %% cell 15
def run_stacking_pipeline(train_path, test_path, best_xgb_params, best_lgb_params, best_cat_params):
    print("1. Data Prep & Deterministic Features...")
    df = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    df['Revenue_log'] = np.log1p(df['Revenue'])
    df['COGS_Ratio'] = df['COGS'] / (df['Revenue'] + 1e-9) 
    
    df = add_features(df)
    df_test = add_features(df_test)

    print(f"Train: {len(df.columns)} features")
    print(f"Test: {len(df_test.columns)} features")
    print()
    
    static_features = [
        'year', 'month', 'day', 'dayofweek',
        "month_sin", "month_cos", 'is_peak_season', 'peak_proximity', 
        'is_sale_season', 'sale_rank', 'days_to_next_sale', 'days_since_last_sale'
    ]
    
    folds = [
        ('2012-07-04', '2018-12-31', '2019-01-01', '2019-12-31'),
        ('2012-07-04', '2019-12-31', '2020-01-01', '2020-12-31'),
        ('2012-07-04', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2012-07-04', '2021-12-31', '2022-01-01', '2022-12-31')
    ]
    
    # Cấu hình Base Models (Tất cả dùng Huber Loss)
    xgb_params = {'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 6, 'objective': 'reg:pseudohubererror', 'tree_method': 'hist', 'random_state': 42}
    lgb_params = {'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 6, 'objective': 'huber', 'random_state': 42, 'verbose': -1}
    cat_params = {'iterations': 2000, 'learning_rate': 0.03, 'depth': 6, 'loss_function': 'Huber:delta=1.5', 'random_state': 42, 'verbose': False}
    
    xgb_params.update(best_xgb_params)
    lgb_params.update(best_lgb_params)
    cat_params.update(best_cat_params)

    print("Base models's params:")
    print(f"best_xgb_params = {xgb_params}")
    print(f"best_lgb_params = {lgb_params}")
    print(f"best_cat_params = {cat_params}")
    print()
    
    # Arrays thu thập dự đoán Out-Of-Fold (OOF) để huấn luyện Trọng tài
    oof_rev_xgb, oof_rev_lgb, oof_rev_cat = [], [], []
    oof_rat_xgb, oof_rat_lgb, oof_rat_cat = [], [], []
    oof_y_rev, oof_y_rat = [], []
    
    print("2. Walk-Forward CV & Generating OOF Predictions...")
    for i, (train_start, train_end, val_start, val_end) in enumerate(folds):
        train_fold = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)].copy()
        val_fold = df[(df['Date'] >= val_start) & (df['Date'] <= val_end)].copy()
        
        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        prophet_df = train_fold[['Date', 'Revenue_log']].rename(columns={'Date': 'ds', 'Revenue_log': 'y'})
        prophet.fit(prophet_df)
        
        train_trend = prophet.predict(prophet_df)['yhat'].values
        val_trend = prophet.predict(val_fold[['Date']].rename(columns={'Date': 'ds'}))['yhat'].values
        train_fold['residual'] = train_fold['Revenue_log'] - train_trend
        val_fold['residual'] = val_fold['Revenue_log'] - val_trend
        
        features = static_features 
        X_tr, X_va = train_fold[features], val_fold[features]
        
        # --- TRAIN BASE MODELS (REVENUE) ---
        m_rev_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        m_rev_xgb.fit(X_tr, train_fold['residual'], eval_set=[(X_va, val_fold['residual'])], verbose=False)
        
        m_rev_lgb = lgb.LGBMRegressor(**lgb_params)
        m_rev_lgb.fit(X_tr, train_fold['residual'], eval_set=[(X_va, val_fold['residual'])], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        m_rev_cat = CatBoostRegressor(**cat_params, early_stopping_rounds=50)
        m_rev_cat.fit(X_tr, train_fold['residual'], eval_set=[(X_va, val_fold['residual'])], verbose=False)
        
        # Thu thập log OOF Revenue 
        log_p_xgb_rev = val_trend + m_rev_xgb.predict(X_va)
        log_p_lgb_rev = val_trend + m_rev_lgb.predict(X_va)
        log_p_cat_rev = val_trend + m_rev_cat.predict(X_va)
        
        oof_rev_xgb.extend(log_p_xgb_rev)
        oof_rev_lgb.extend(log_p_lgb_rev)
        oof_rev_cat.extend(log_p_cat_rev)
        oof_y_rev.extend(val_fold['Revenue_log'])
        
        # --- TRAIN BASE MODELS (COGS RATIO) ---
        m_rat_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        m_rat_xgb.fit(X_tr, train_fold['COGS_Ratio'], eval_set=[(X_va, val_fold['COGS_Ratio'])], verbose=False)
        
        m_rat_lgb = lgb.LGBMRegressor(**lgb_params)
        m_rat_lgb.fit(X_tr, train_fold['COGS_Ratio'], eval_set=[(X_va, val_fold['COGS_Ratio'])], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        m_rat_cat = CatBoostRegressor(**cat_params, early_stopping_rounds=50)
        m_rat_cat.fit(X_tr, train_fold['COGS_Ratio'], eval_set=[(X_va, val_fold['COGS_Ratio'])], verbose=False)
        
        # Thu thập OOF Ratio
        oof_rat_xgb.extend(m_rat_xgb.predict(X_va))
        oof_rat_lgb.extend(m_rat_lgb.predict(X_va))
        oof_rat_cat.extend(m_rat_cat.predict(X_va))
        oof_y_rat.extend(val_fold['COGS_Ratio'])

    
    print("3. Training Meta-Models (HuberRegressor)...")
    X_meta_rev = np.column_stack([oof_rev_xgb, oof_rev_lgb, oof_rev_cat])
    # Revenue Meta Tuning
    huber_grid = {'epsilon': [1.2, 1.25, 1.35, 1.4, 1.45, 1.5], 'alpha': [0.0001, 0.001, 0.01, 0.1]}
    meta_rev_search = GridSearchCV(HuberRegressor(fit_intercept=False), huber_grid, cv=5, scoring='neg_mean_absolute_error')
    meta_rev_search.fit(X_meta_rev, oof_y_rev)
    meta_rev_model = meta_rev_search.best_estimator_
    
    X_meta_rat = np.column_stack([oof_rat_xgb, oof_rat_lgb, oof_rat_cat])
    # Ratio Meta Tuning (Epsilon nhỏ hơn vì khoảng giá trị nhỏ)
    huber_ratio_grid = {'epsilon': [1.01, 1.05, 1.1, 1.2], 'alpha': [0.0001, 0.001, 0.01, 0.1]}
    meta_rat_search = GridSearchCV(HuberRegressor(fit_intercept=False), huber_ratio_grid, cv=5, scoring='neg_mean_absolute_error')
    meta_rat_search.fit(X_meta_rat, oof_y_rat)
    meta_rat_model = meta_rat_search.best_estimator_
    
    print(f" -> Trọng số (XGB/LGB/CAT) do Meta Model cấp - Revenue: {meta_rev_model.coef_}")
    print(f" -> Trọng số (XGB/LGB/CAT) do Meta Model cấp - Ratio: {meta_rat_model.coef_}")

    
    print("4. Training Full Base Models for Test Set...")
    
    final_features = static_features
    
    # Bỏ early_stopping
    xgb_params.pop('early_stopping_rounds', None)
    cat_params.pop('early_stopping_rounds', None)
    xgb_params['n_estimators'] = 600
    lgb_params['n_estimators'] = 600
    cat_params['iterations'] = 600
    
    final_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_df_full = df[['Date', 'Revenue_log']].rename(columns={'Date': 'ds', 'Revenue_log': 'y'})
    final_prophet.fit(prophet_df_full)
    df['residual'] = df['Revenue_log'] - final_prophet.predict(prophet_df_full)['yhat'].values
    
    # Train Revenue Full
    f_rev_xgb, f_rev_lgb, f_rev_cat = xgb.XGBRegressor(**xgb_params), lgb.LGBMRegressor(**lgb_params), CatBoostRegressor(**cat_params)
    f_rev_xgb.fit(df[final_features], df['residual'], verbose=False)
    f_rev_lgb.fit(df[final_features], df['residual'])
    f_rev_cat.fit(df[final_features], df['residual'], verbose=False)
    
    # Train Ratio Full
    f_rat_xgb, f_rat_lgb, f_rat_cat = xgb.XGBRegressor(**xgb_params), lgb.LGBMRegressor(**lgb_params), CatBoostRegressor(**cat_params)
    f_rat_xgb.fit(df[final_features], df['COGS_Ratio'], verbose=False)
    f_rat_lgb.fit(df[final_features], df['COGS_Ratio'])
    f_rat_cat.fit(df[final_features], df['COGS_Ratio'], verbose=False)
    
    print("5. Generating Meta-Predictions...")
    test_trend = final_prophet.predict(df_test[['Date']].rename(columns={'Date': 'ds'}))['yhat'].values
    
    # Base Predictions (Revenue)
    test_log_p_rev_xgb = test_trend + f_rev_xgb.predict(df_test[final_features])
    test_log_p_rev_lgb = test_trend + f_rev_lgb.predict(df_test[final_features])
    test_log_p_rev_cat = test_trend + f_rev_cat.predict(df_test[final_features])
    
    # Base Predictions (Ratio)
    test_p_rat_xgb = f_rat_xgb.predict(df_test[final_features])
    test_p_rat_lgb = f_rat_lgb.predict(df_test[final_features])
    test_p_rat_cat = f_rat_cat.predict(df_test[final_features])
    
    # Meta Model Inference
    X_test_meta_rev = np.column_stack([test_log_p_rev_xgb, test_log_p_rev_lgb, test_log_p_rev_cat])
    X_test_meta_rat = np.column_stack([test_p_rat_xgb, test_p_rat_lgb, test_p_rat_cat])
    
    final_log_rev = meta_rev_model.predict(X_test_meta_rev)
    df_test['Revenue'] = np.expm1(final_log_rev)
    df_test['COGS_Ratio_Pred'] = meta_rat_model.predict(X_test_meta_rat)
    
    df_test['COGS'] = df_test['Revenue'] * df_test['COGS_Ratio_Pred']
    df_test['Revenue'] = df_test['Revenue'].clip(lower=0)
    df_test['COGS'] = df_test['COGS'].clip(lower=0)
    
    submission_name = "submission.csv"
    submission = df_test[['Date', 'Revenue', 'COGS']]
    submission.to_csv(submission_name, index=False)
    print(f"Done! Saved to {submission_name}")

# %% cell 16
run_stacking_pipeline(
    DIR + "sales.csv", 
    DIR + "sample_submission.csv", 
    best_xgb_params, best_lgb_params, best_cat_params
)
