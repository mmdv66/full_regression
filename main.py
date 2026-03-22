import yaml
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import train_test_split, cross_val_predict, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import warnings
warnings.filterwarnings("ignore")


# ── Data 

def load_data(cfg):
    df_train = pd.read_csv(cfg["data"]["train_path"])
    df_test  = pd.read_csv(cfg["data"]["test_path"])
    return df_train, df_test


def preprocess(df_train, df_test, cfg):
    """Удаляет выбросы, заполняет пропуски, разделяет X/y."""
    target = cfg["preprocessing"]["target"]

    df_train = df_train.drop(
        df_train[(df_train["GrLivArea"] > 4000) & (df_train[target] < 300_000)].index
    )

    y = df_train[target]
    X = df_train.drop(columns=["Id", target], errors="ignore")

    n = cfg["preprocessing"]["missing_threshold"]
    missing = X.isnull().sum().sort_values(ascending=False)
    drop_cols = list(missing[:n].index)

    X        = X.drop(columns=drop_cols, errors="ignore")
    df_test  = df_test.drop(columns=["Id"] + drop_cols, errors="ignore")

    X       = X.apply(lambda col: col.fillna(col.value_counts().index[0]))
    df_test = df_test.apply(lambda col: col.fillna(col.value_counts().index[0]))

    return X, y, df_test


def build_preprocessor(X):
    """Строит ColumnTransformer для числовых, категориальных и порядковых признаков."""
    cat_features = X.select_dtypes(include="object").columns.tolist()
    num_features = [c for c in X.columns if X[c].nunique() > 15  and X[c].dtype in ["float64", "int64"]]
    ord_features = [c for c in X.columns if X[c].nunique() <= 15 and X[c].dtype in ["float64", "int64"]]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),                                               num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),                         cat_features),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ord_features),
    ], remainder="passthrough")

    return preprocessor


# Metrics 

def print_rmse(name, y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"{name:<35} RMSE: {rmse:.2f}")
    return rmse


# Sklearn models 

def train_sklearn_models(X_tr, y_tr_log, X_val, y_val, cfg):
    """Обучает все sklearn/boosting модели, возвращает предсказания и RMSE."""
    rs = cfg["training"]["random_state"]
    cv = cfg["training"]["cv_folds"]
    results = {}

    def _fit_predict(name, model):
        model.fit(X_tr, y_tr_log)
        pred = np.expm1(model.predict(X_val))
        results[name] = {"model": model, "pred": pred, "rmse": print_rmse(name, y_val, pred)}
        return model, pred

    # Baseline
    _fit_predict("LinearRegression", LinearRegression())
    _fit_predict("Lasso",            Lasso(alpha=cfg["models"]["lasso"]["alpha"], random_state=rs))
    _fit_predict("Ridge",            Ridge(alpha=cfg["models"]["ridge"]["alpha"], random_state=rs))

    # ElasticNet GridSearch
    gs_en = GridSearchCV(
        ElasticNet(random_state=rs, max_iter=1000),
        {"alpha": cfg["models"]["elasticnet"]["alphas"], "l1_ratio": cfg["models"]["elasticnet"]["l1_ratios"]},
        cv=cv, scoring="neg_mean_squared_error", n_jobs=-1,
    )
    _fit_predict("ElasticNet", gs_en)

    # KNN RandomSearch
    rs_knn = RandomizedSearchCV(
        KNeighborsRegressor(),
        {"n_neighbors": randint(1, 30), "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
        n_iter=cfg["models"]["knn"]["n_iter"], cv=cv, scoring="neg_mean_squared_error",
        n_jobs=-1, random_state=rs,
    )
    _fit_predict("KNN", rs_knn)

    # RandomForest RandomSearch
    rs_rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=rs),
        {"max_depth": randint(4, 15), "min_samples_split": randint(2, 20), "min_samples_leaf": randint(1, 20)},
        n_iter=cfg["models"]["random_forest"]["n_iter"], cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=rs,
    )
    _fit_predict("RandomForest", rs_rf)

    # CatBoost GridSearch
    gs_cb = GridSearchCV(
        CatBoostRegressor(loss_function="RMSE", verbose=0, random_state=rs, allow_writing_files=False),
        {k: v for k, v in cfg["models"]["catboost"].items()},
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    _fit_predict("CatBoost", gs_cb)

    # LightGBM RandomSearch
    rs_lgbm = RandomizedSearchCV(
        LGBMRegressor(random_state=rs, verbose=-1),
        {
            "n_estimators": randint(100, 1000), "learning_rate": uniform(0.005, 0.2),
            "max_depth": randint(3, 15), "num_leaves": randint(10, 255),
            "reg_alpha": loguniform(1e-4, 10), "reg_lambda": loguniform(1e-4, 10),
            "subsample": uniform(0.5, 0.5), "colsample_bytree": uniform(0.5, 0.5),
        },
        n_iter=cfg["models"]["lightgbm"]["n_iter"], cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=rs,
    )
    _fit_predict("LightGBM", rs_lgbm)

    # XGBoost GridSearch
    gs_xgb = GridSearchCV(
        XGBRegressor(random_state=rs, objective="reg:squarederror"),
        {k: v for k, v in cfg["models"]["xgboost"].items()},
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    _fit_predict("XGBoost", gs_xgb)

    return results


#Neural network 

class HousePriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImprovedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(0.1), nn.Dropout(0.15),
            nn.Linear(64, 32),        nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


def train_nn(X_tr, y_tr_log, X_val, y_val_log, cfg):
    """Обучает ImprovedNN, возвращает предсказания на val в исходном масштабе."""
    X_tr_dense  = X_tr.toarray()  if hasattr(X_tr,  "toarray") else X_tr
    X_val_dense = X_val.toarray() if hasattr(X_val, "toarray") else X_val

    train_loader = DataLoader(
        HousePriceDataset(X_tr_dense, y_tr_log.values),
        batch_size=cfg["nn"]["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        HousePriceDataset(X_val_dense, y_val_log.values),
        batch_size=cfg["nn"]["batch_size"], shuffle=False,
    )

    model     = ImprovedNN(X_tr_dense.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["nn"]["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    criterion = nn.MSELoss()

    for epoch in range(cfg["nn"]["epochs_improved"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = [criterion(model(xb), yb).item() for xb, yb in val_loader]
        scheduler.step(np.mean(val_losses))

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{cfg['nn']['epochs_improved']}  val_loss={np.mean(val_losses):.4f}")

    model.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val_dense)
        pred_log = model(X_val_t).numpy()

    return np.expm1(pred_log)


#  Stacking 

def train_stacking(sklearn_results, X_tr, y_tr_log, y_val, cfg):
    """OOF стекинг поверх sklearn моделей."""
    base_models = [v["model"] for v in sklearn_results.values()]
    oof_preds   = np.column_stack([
        cross_val_predict(m, X_tr, y_tr_log, cv=cfg["training"]["cv_folds"])
        for m in base_models
    ])

    meta = Ridge(alpha=1.0, random_state=cfg["training"]["random_state"])
    meta.fit(oof_preds, y_tr_log)

    val_preds  = np.column_stack([v["pred"] for v in sklearn_results.values()])
    val_preds_log = np.log1p(val_preds)
    stacking_pred = np.expm1(meta.predict(val_preds_log))

    print_rmse("Stacking", y_val, stacking_pred)
    return stacking_pred


#Main 

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=== Загрузка данных ===")
    df_train, df_test = load_data(cfg)

    print("=== Препроцессинг ===")
    X, y, df_test = preprocess(df_train, df_test, cfg)

    rs   = cfg["training"]["random_state"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg["training"]["test_size"], random_state=rs
    )

    preprocessor  = build_preprocessor(X_tr)
    X_tr_proc     = preprocessor.fit_transform(X_tr)
    X_val_proc    = preprocessor.transform(X_val)

    y_tr_log  = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    print("\n=== Обучение sklearn/boosting моделей ===")
    sklearn_results = train_sklearn_models(X_tr_proc, y_tr_log, X_val_proc, y_val, cfg)

    print("\n=== Обучение нейросети ===")
    nn_pred = train_nn(X_tr_proc, y_tr_log, X_val_proc, y_val_log, cfg)
    print_rmse("ImprovedNN", y_val, nn_pred)

    print("\n=== Стекинг ===")
    train_stacking(sklearn_results, X_tr_proc, y_tr_log, y_val, cfg)

    print("\n=== Итоговая таблица RMSE ===")
    all_rmse = {name: v["rmse"] for name, v in sklearn_results.items()}
    all_rmse["ImprovedNN"] = root_mean_squared_error(y_val, nn_pred)
    summary = pd.DataFrame(all_rmse.items(), columns=["Model", "RMSE"]).sort_values("RMSE")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()