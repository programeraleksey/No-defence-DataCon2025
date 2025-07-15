from __future__ import annotations
import os, random, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential

"""На данном этапе мы начинаем обучение и сравнение разных моделей по ранее заготовленному датасету. 
Для работы требуется DataSet_with_2D.csv"""

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

warnings.filterwarnings('ignore')

"""## Загрузка и первичный анализ данных"""

DATA_PATH = Path('../../../AI/Task 4/DataSet_with_2D.csv')
assert DATA_PATH.exists(), f'Файл {DATA_PATH} не найден. Поместите CSV рядом с ноутбуком.'

df = pd.read_csv(DATA_PATH)

# Приводим столбец IC50 к числовому виду и убираем некорректные/отрицательные значения
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df[df['Value'] > 0].dropna(subset=['Value'])

print(f'Размерность набора: {df.shape}')

"""Предобработка данных
Применяем преобразование IC₅₀ → pIC₅₀, выбираем только числовые признаки, убираем некоректные данные.
"""

def to_pic50(nm: float) -> float:
    """Перевод IC50 (нМ) → pIC50"""
    return 9.0 - math.log10(nm)

y = df['Value'].apply(to_pic50).values

# Отбрасываем нечисловые признаки (SMILES и т.п.)
X_tab = (
    df.drop(columns=['Value', 'Smiles'], errors='ignore')
      .select_dtypes(include=[np.number])
)
# Чистка NaN/inf
X_tab.replace([np.inf, -np.inf], np.nan, inplace=True)
X_tab.dropna(axis=1, inplace=True)   # убираем признаки с пропусками
X_tab = X_tab.clip(-np.finfo(np.float32).max, np.finfo(np.float32).max).astype(np.float32).values

print('Форма матрицы признаков:', X_tab.shape)

"""## Метрики и вспомогательные функции"""

def metric_pack(y_true, y_pred):
    """Возвращает (R², RMSE, MAE)"""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return r2_score(y_true, y_pred), rmse, mean_absolute_error(y_true, y_pred)

import pandas as pd

"""## Конфигурации моделей"""

# Гиперпараметры
cfg_rf = dict(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=1,
    random_state=SEED,
    n_jobs=-1
)

cfg_lgb = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=256,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    objective='regression',
    verbose=-1
)

cfg_mlp = dict(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    alpha=1e-3,
    max_iter=400,
    early_stopping=True,
    random_state=SEED
)

CNN_EPOCHS = 100
CNN_BATCH  = 32

def build_cnn(shape: tuple[int, int]) -> Sequential:
    model = Sequential([
        InputLayer(shape=shape),
        Conv1D(64, 3, padding='same', activation='relu'),
        Conv1D(32, 3, padding='same', activation='relu'),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model

"""## Кросс‑валидация и сравнение моделей
Используем **5‑кратную** кросс‑валидацию (`KFold`) со случайным перемешиванием и фиксированным зерном.
"""

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

results = {k: [] for k in ['RF', 'LGBM', 'MLP', 'CNN']}

for fold, (train_idx, test_idx) in enumerate(kf.split(X_tab), 1):
    X_tr, X_te = X_tab[train_idx], X_tab[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Random Forest
    rf = RandomForestRegressor(**cfg_rf).fit(X_tr, y_tr)
    results['RF'].append(metric_pack(y_te, rf.predict(X_te)))

    # LightGBM
    lgbm = LGBMRegressor(**cfg_lgb).fit(X_tr, y_tr)
    results['LGBM'].append(metric_pack(y_te, lgbm.predict(X_te)))

    # MLP
    mlp = Pipeline([('s', StandardScaler()), ('m', MLPRegressor(**cfg_mlp))])
    mlp.fit(X_tr, y_tr)
    results['MLP'].append(metric_pack(y_te, mlp.predict(X_te)))

    # 1‑D CNN
    scaler = StandardScaler().fit(X_tr)
    Xt_s, Xe_s = scaler.transform(X_tr)[..., None], scaler.transform(X_te)[..., None]
    cnn = build_cnn((Xt_s.shape[1], 1))
    cnn.fit(
        Xt_s, y_tr,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH,
        validation_split=0.2,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)],
    )
    results['CNN'].append(metric_pack(y_te, cnn.predict(Xe_s, verbose=0).flatten()))

    print(f'Fold {fold} завершён')

# Сводная таблица
import pandas as pd
rows = []
for name, vals in results.items():
    r2s, rmses, maes = zip(*vals)
    rows.append(dict(
        Model=name,
        R2_mean=np.mean(r2s), R2_std=np.std(r2s),
        RMSE_mean=np.mean(rmses), RMSE_std=np.std(rmses),
        MAE_mean=np.mean(maes), MAE_std=np.std(maes),
    ))
cv_summary = pd.DataFrame(rows).set_index('Model').round(3)

"""### Визуализация результатов"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7,4))
cv_summary['R2_mean'].plot(kind='bar', ax=ax)
ax.set_title('Сравнение моделей по R² (среднее по фолдам)')
ax.set_ylabel('R²')
ax.set_xlabel('Модель')
ax.grid(True, axis='y', lw=0.3, ls='--')
plt.show()

"""#### Выводы
- **Random Forest** и **LightGBM** демонстрирует наивысший средний R² и наименьшие ошибку RMSE/MAE, что делает их лучшим выбором для данного набора дескрипторов.
- **MLP** справляется с заданием хуже, но тоже на приемлемом уровне.
- **CNN** показывает результаты в разы хуже, поэтому ее использование для нашего датасета бесполезно.

## Обучение финальных моделей на всей выборке и сохранение
"""

OUT_DIR = Path('models')
OUT_DIR.mkdir(exist_ok=True)

# Random Forest
rf_full = RandomForestRegressor(**cfg_rf).fit(X_tab, y)
joblib.dump(rf_full, OUT_DIR / 'rf.joblib')

# LightGBM
lgbm_full = LGBMRegressor(**cfg_lgb).fit(X_tab, y)
joblib.dump(lgbm_full, OUT_DIR / 'lgbm.joblib')

# MLP
mlp_full = Pipeline([('s', StandardScaler()), ('m', MLPRegressor(**cfg_mlp))])
mlp_full.fit(X_tab, y)
joblib.dump(mlp_full, OUT_DIR / 'mlp.joblib')

# CNN
scaler_all = StandardScaler().fit(X_tab)
X_all_scaled = scaler_all.transform(X_tab)[..., None]
cnn_full = build_cnn((X_all_scaled.shape[1], 1))
cnn_full.fit(
    X_all_scaled, y,
    epochs=CNN_EPOCHS,
    batch_size=CNN_BATCH,
    validation_split=0.2,
    verbose=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0)],
)
cnn_full.save(OUT_DIR / 'cnn.keras')
joblib.dump(scaler_all, OUT_DIR / 'cnn_scaler.joblib')

print('Все модели сохранены в', OUT_DIR.resolve())