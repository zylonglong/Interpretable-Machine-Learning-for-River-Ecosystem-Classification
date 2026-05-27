
# ============================================================
# LightGBM Optimized Full Pipeline (data-only version)
# - Outer loop: 5 repeated stratified splits (80/20) with seeds 42..46
# - Inner loop: RandomizedSearchCV + 5-fold StratifiedKFold (tuning)
# - Evaluation: export confusion matrix / ROC / prediction data only
# - Learning curve: export mean±std + raw export + JSON only
# - SHAP: export long-format data, feature importance, JSON only
# - Aggregation: merge 5 folds, compute mean±sd, export aggregated files + json
# ============================================================

import os
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import joblib

from tqdm import tqdm
from lightgbm import LGBMClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, label_binarize


warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "font.family": "Arial",
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "legend.fontsize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.titlesize": 24,
    "figure.titlesize": 24
})


@dataclass(frozen=True)
class Config:
    data_excel_path: str = r"E:\桌面\水生态项目文件\深圳源清项目\2025.5文章数据整理\RobustScaler标准化结果.xlsx"
    sheet_name: str = "36D_Feature_Matrix"
    base_output_root: str = r"E:\桌面\水生态项目文件\深圳源清项目\2025.5文章数据整理\1220分组预测结果可视化"
    model_name: str = "LightGBM"
    outer_seeds: tuple = (42, 43, 44, 45, 46)
    test_size: float = 0.2
    inner_cv_splits: int = 5
    random_search_iter: int = 50
    shap_samples_per_page: int = 20


CFG = Config()

TROPHIC_GROUPS = {
    "Phytoplankton": ["P_"],
    "PeriphyticAlgae": ["PA_"],
    "BenthicMacroinvertebrates": ["BM_"],
    "Zooplankton": ["Z_"],
    "AquaticPlants": ["AP_"],
    "Fish": ["F_"]
}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return obj


def infer_group_from_feature(feature: str) -> str:
    for group, prefixes in TROPHIC_GROUPS.items():
        for p in prefixes:
            if feature.startswith(p):
                return group
    return "Other"


def target_out_dir(target_column: str) -> str:
    t = target_column.replace(" ", "_")
    return ensure_dir(os.path.join(CFG.base_output_root, t, CFG.model_name))


def fold_out_dir(base_target_out: str, fold_id: int) -> str:
    return ensure_dir(os.path.join(base_target_out, f"fold_{fold_id}"))


def agg_out_dir(base_target_out: str) -> str:
    return ensure_dir(os.path.join(base_target_out, "aggregated_shap_results"))


def load_data(target_column: str):
    df = pd.read_excel(CFG.data_excel_path, sheet_name=CFG.sheet_name)
    X = df.filter(regex="_st$").astype(np.float32)
    y = df[target_column]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    feature_names = X.columns.tolist()
    class_names = le.classes_
    return X, y_enc, le, feature_names, class_names


def get_lgb_param_grid():
    return {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [-1, 3, 5, 7, 10],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [0.1, 1, 10],
        "min_split_gain": [0.0, 0.1, 0.5, 1.0],
    }


def build_lgb_base_model(n_classes: int):
    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )


def tune_hyperparameters(X_train, y_train, n_classes: int):
    param_grid = get_lgb_param_grid()
    base_model = build_lgb_base_model(n_classes)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    rs = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=CFG.random_search_iter,
        cv=StratifiedKFold(n_splits=CFG.inner_cv_splits, shuffle=True, random_state=42),
        scoring=scoring,
        refit="accuracy",
        n_jobs=1,
        verbose=2,
        random_state=42,
        return_train_score=True,
    )
    print("\n[LightGBM] 开始 RandomizedSearchCV 超参数调优 ...")
    rs.fit(X_train, y_train)
    print("[LightGBM] 调参完成。best_score =", rs.best_score_)
    return rs.best_estimator_, rs


def plot_hyperparameter_search_results(rs: RandomizedSearchCV, fold_dir_path: str):
    hp_dir = ensure_dir(os.path.join(fold_dir_path, f"{CFG.model_name}_hyperparameters"))

    results = pd.DataFrame(rs.cv_results_)
    results.to_excel(os.path.join(hp_dir, "hyperparameter_search_results.xlsx"), index=False)
    results.to_csv(os.path.join(hp_dir, "hyperparameter_search_results.csv"), index=False)

    params = [c for c in results.columns if c.startswith("param_")]
    metrics = ["mean_test_accuracy", "mean_test_precision", "mean_test_recall", "mean_test_f1"]

    results_clean = results.dropna(subset=["mean_test_accuracy"]).copy()

    for p in params:
        try:
            export_cols = [p] + [m for m in metrics if m in results_clean.columns]
            param_df = results_clean[export_cols].copy()
            pname = p.replace("param_", "")
            param_df.to_excel(os.path.join(hp_dir, f"{pname}_plot_data.xlsx"), index=False)
            param_df.to_csv(os.path.join(hp_dir, f"{pname}_plot_data.csv"), index=False)
        except Exception:
            pass

    try:
        from sklearn.ensemble import RandomForestRegressor

        Xp = results_clean[params].copy().fillna(0)
        y_score = results_clean["mean_test_accuracy"].copy()

        for col in Xp.columns:
            if Xp[col].dtype == "object":
                Xp[col] = Xp[col].astype(str).astype("category").cat.codes

        rfr = RandomForestRegressor(n_estimators=200, random_state=42)
        rfr.fit(Xp, y_score)

        imp = pd.DataFrame({
            "Parameter": [p.replace("param_", "") for p in params],
            "Importance": rfr.feature_importances_,
        }).sort_values("Importance", ascending=False)

        imp.to_excel(os.path.join(hp_dir, "hyperparameter_importance.xlsx"), index=False)
        imp.to_csv(os.path.join(hp_dir, "hyperparameter_importance.csv"), index=False)
    except Exception:
        pass

    return hp_dir


def plot_model_evaluation(model, X_test, y_test, class_names, fold_dir_path: str):
    eval_dir = ensure_dir(os.path.join(fold_dir_path, f"{CFG.model_name}_evaluation"))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "True_Label"
    cm_df.columns.name = "Predicted_Label"
    cm_df.to_excel(os.path.join(eval_dir, "confusion_matrix_data.xlsx"))
    cm_df.to_csv(os.path.join(eval_dir, "confusion_matrix_data.csv"))

    cm_long_records = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            cm_long_records.append({
                "True_Label": true_name,
                "Predicted_Label": pred_name,
                "Count": int(cm[i, j])
            })
    cm_long_df = pd.DataFrame(cm_long_records)
    cm_long_df.to_excel(os.path.join(eval_dir, "confusion_matrix_long_data.xlsx"), index=False)
    cm_long_df.to_csv(os.path.join(eval_dir, "confusion_matrix_long_data.csv"), index=False)

    y_bin = label_binarize(y_test, classes=np.unique(y_test))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)
    n_classes = y_bin.shape[1]

    roc_records = []
    auc_records = []
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_records.append({"Class": class_names[i], "AUC": roc_auc})
        for k in range(len(fpr)):
            roc_records.append({
                "Class": class_names[i],
                "FPR": float(fpr[k]),
                "TPR": float(tpr[k]),
                "Threshold": float(thresholds[k]),
                "AUC": float(roc_auc)
            })

    roc_df = pd.DataFrame(roc_records)
    roc_df.to_excel(os.path.join(eval_dir, "roc_curve_raw_data.xlsx"), index=False)
    roc_df.to_csv(os.path.join(eval_dir, "roc_curve_raw_data.csv"), index=False)

    auc_df = pd.DataFrame(auc_records)
    auc_df.to_excel(os.path.join(eval_dir, "roc_auc_summary.xlsx"), index=False)
    auc_df.to_csv(os.path.join(eval_dir, "roc_auc_summary.csv"), index=False)

    prob_df = pd.DataFrame({
        "True_Label_Encoded": np.array(y_test).reshape(-1),
        "Predicted_Label_Encoded": np.array(y_pred).reshape(-1),
    })
    for i, cls in enumerate(class_names):
        prob_df[f"Prob_{cls}"] = y_proba[:, i]
    prob_df.to_excel(os.path.join(eval_dir, "prediction_probabilities.xlsx"), index=False)
    prob_df.to_csv(os.path.join(eval_dir, "prediction_probabilities.csv"), index=False)

    return eval_dir


def plot_learning_curve(model_params: dict, X_train, y_train, fold_dir_path: str, n_classes: int):
    lc_dir = ensure_dir(os.path.join(fold_dir_path, f"{CFG.model_name}_learning_curve"))

    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    train_fracs = np.unique(np.linspace(0.1, 1.0, 20, endpoint=True))

    records = []
    fold_counter = 0
    for frac in tqdm(train_fracs, desc=f"[{CFG.model_name}] Learning curve"):
        fold_counter = 0
        for tr_idx, val_idx in splitter.split(X_train, y_train):
            fold_counter += 1
            n = max(1, int(frac * len(tr_idx)))
            tr_sub = tr_idx[:n]

            mdl = LGBMClassifier(**model_params)
            mdl.set_params(objective="multiclass", num_class=n_classes, verbose=-1)
            mdl.fit(X_train.iloc[tr_sub], y_train[tr_sub])

            records.append({
                "train_size": int(n),
                "fold": int(fold_counter),
                "train_score": float(mdl.score(X_train.iloc[tr_sub], y_train[tr_sub])),
                "val_score": float(mdl.score(X_train.iloc[val_idx], y_train[val_idx])),
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(lc_dir, "learning_curve_fold_details.csv"), index=False)
    df.to_excel(os.path.join(lc_dir, "learning_curve_fold_details.xlsx"), index=False)

    agg = df.groupby("train_size").agg(
        train_mean=("train_score", "mean"),
        train_std=("train_score", "std"),
        val_mean=("val_score", "mean"),
        val_std=("val_score", "std"),
    ).reset_index().sort_values("train_size")

    agg.to_csv(os.path.join(lc_dir, "learning_curve_summary.csv"), index=False)
    agg.to_excel(os.path.join(lc_dir, "learning_curve_summary.xlsx"), index=False)

    out_json = {
        "train_sizes": to_serializable(agg["train_size"].values),
        "train_mean": to_serializable(agg["train_mean"].values),
        "train_std": to_serializable(agg["train_std"].values),
        "val_mean": to_serializable(agg["val_mean"].values),
        "val_std": to_serializable(agg["val_std"].values),
    }
    with open(os.path.join(lc_dir, "learning_curve_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    return lc_dir


def _normalize_expected_value(expected_value, n_classes: int):
    if isinstance(expected_value, list):
        ev = np.array(expected_value, dtype=float)
        if ev.size == n_classes:
            return ev
    if isinstance(expected_value, (float, np.floating, int, np.integer)):
        return np.array([float(expected_value)] * n_classes, dtype=float)
    try:
        ev = np.array(expected_value, dtype=float)
        if ev.size == n_classes:
            return ev
    except Exception:
        pass
    return np.array([0.0] * n_classes, dtype=float)


def build_shap_long_df(shap_values_list, X_test: pd.DataFrame, feature_names, class_names, fold_id: int):
    n_samples = X_test.shape[0]
    all_records = []

    for c_idx, cls in enumerate(class_names):
        if c_idx >= len(shap_values_list):
            break
        sv = shap_values_list[c_idx]
        for i in range(n_samples):
            fv = X_test.iloc[i].values
            for f_idx, feat in enumerate(feature_names):
                all_records.append({
                    "Fold": fold_id,
                    "Group": infer_group_from_feature(feat),
                    "Class": str(cls),
                    "Feature": feat,
                    "ShapValue": float(sv[i, f_idx]),
                    "FeatureValue": float(fv[f_idx]),
                    "SampleIndex": int(i),
                })

    return pd.DataFrame(all_records)


def shap_analysis_single(model, X_test, y_test, feature_names, class_names, fold_id: int, fold_dir_path: str):
    shap_dir = ensure_dir(os.path.join(fold_dir_path, f"{CFG.model_name}_shap"))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    expected_value = _normalize_expected_value(explainer.expected_value, n_classes=len(class_names))

    if isinstance(shap_values, np.ndarray):
        shap_values_list = [shap_values]
    else:
        shap_values_list = shap_values

    long_df = build_shap_long_df(shap_values_list, X_test, feature_names, class_names, fold_id)
    long_df.to_csv(os.path.join(shap_dir, "beeswarm_data.csv"), index=False)
    long_df.to_excel(os.path.join(shap_dir, "beeswarm_data.xlsx"), index=False)
    long_df.to_csv(os.path.join(shap_dir, "dependence_data.csv"), index=False)
    long_df.to_excel(os.path.join(shap_dir, "dependence_data.xlsx"), index=False)

    per_class_importance = {}
    for c_idx, cls in enumerate(class_names):
        if c_idx >= len(shap_values_list):
            break
        per_class_importance[str(cls)] = np.mean(np.abs(shap_values_list[c_idx]), axis=0)

    if len(shap_values_list) > 1:
        global_importance = np.mean(np.abs(np.stack(shap_values_list, axis=0)), axis=(0, 1))
    else:
        global_importance = np.mean(np.abs(shap_values_list[0]), axis=0)

    imp_df = pd.DataFrame({"Feature": feature_names})
    for cls, imp in per_class_importance.items():
        imp_df[f"Importance_{cls}"] = imp
    imp_df["Global_Importance"] = global_importance
    imp_df.sort_values("Global_Importance", ascending=False, inplace=True)
    imp_df.to_excel(os.path.join(shap_dir, "feature_importance.xlsx"), index=False)
    imp_df.to_csv(os.path.join(shap_dir, "feature_importance.csv"), index=False)

    shap_summary = {
        "fold": fold_id,
        "model": CFG.model_name,
        "n_samples_test": int(X_test.shape[0]),
        "class_names": [str(c) for c in class_names],
        "expected_value": to_serializable(expected_value),
        "global_importance": to_serializable(global_importance),
        "per_class_importance": {k: to_serializable(v) for k, v in per_class_importance.items()},
    }
    with open(os.path.join(shap_dir, "shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(shap_summary, f, indent=2)

    return shap_dir


def aggregate_shap_results(base_target_out: str):
    agg_dir_path = agg_out_dir(base_target_out)

    beeswarm_all = []
    dep_all = []
    imp_all = []

    for fold_id in range(1, len(CFG.outer_seeds) + 1):
        fdir = os.path.join(base_target_out, f"fold_{fold_id}", f"{CFG.model_name}_shap")
        beeswarm_path = os.path.join(fdir, "beeswarm_data.csv")
        dep_path = os.path.join(fdir, "dependence_data.csv")
        imp_path = os.path.join(fdir, "feature_importance.xlsx")

        if os.path.exists(beeswarm_path):
            beeswarm_all.append(pd.read_csv(beeswarm_path))
        if os.path.exists(dep_path):
            dep_all.append(pd.read_csv(dep_path))
        if os.path.exists(imp_path):
            tmp = pd.read_excel(imp_path)
            tmp["Fold"] = fold_id
            imp_all.append(tmp)

    if not beeswarm_all or not imp_all:
        print(f"[{CFG.model_name}] aggregate_shap_results: 未找到 fold SHAP 输出，跳过聚合。")
        return None

    combined_beeswarm = pd.concat(beeswarm_all, ignore_index=True)
    combined_dependence = pd.concat(dep_all, ignore_index=True) if dep_all else combined_beeswarm.copy()

    combined_beeswarm.to_csv(os.path.join(agg_dir_path, "combined_beeswarm_data.csv"), index=False)
    combined_beeswarm.to_excel(os.path.join(agg_dir_path, "combined_beeswarm_data.xlsx"), index=False)
    combined_dependence.to_csv(os.path.join(agg_dir_path, "combined_dependence_data.csv"), index=False)
    combined_dependence.to_excel(os.path.join(agg_dir_path, "combined_dependence_data.xlsx"), index=False)

    imp_all_df = pd.concat(imp_all, ignore_index=True)

    global_stats = (
        imp_all_df[["Feature", "Global_Importance"]]
        .groupby("Feature")["Global_Importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .rename(columns={"mean": "Global_Mean", "std": "Global_SD"})
    )

    class_cols = [c for c in imp_all_df.columns if c.startswith("Importance_")]
    merged = global_stats
    for c in class_cols:
        st = (
            imp_all_df[["Feature", c]]
            .groupby("Feature")[c]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": f"{c}_Mean", "std": f"{c}_SD"})
        )
        merged = merged.merge(st, on="Feature", how="left")

    merged.to_excel(os.path.join(agg_dir_path, "aggregated_feature_importance.xlsx"), index=False)
    merged.to_csv(os.path.join(agg_dir_path, "aggregated_feature_importance.csv"), index=False)

    summary_json = {
        "model": CFG.model_name,
        "n_folds": len(CFG.outer_seeds),
        "top20_global": merged.head(20).to_dict(orient="records")
    }
    with open(os.path.join(agg_dir_path, "aggregated_shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_serializable(summary_json), f, indent=2)

    return agg_dir_path


def run_single_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    le: LabelEncoder,
    feature_names,
    class_names,
    base_target_out: str,
    fold_id: int,
    seed: int,
):
    f_out = fold_out_dir(base_target_out, fold_id)

    idx_all = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx_all,
        test_size=CFG.test_size,
        stratify=y,
        random_state=seed
    )

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    model, search = tune_hyperparameters(X_train, y_train, n_classes=len(class_names))

    model_path = os.path.join(f_out, f"{CFG.model_name}.pkl")
    joblib.dump(model, model_path)

    best_params = to_serializable(search.best_params_)
    with open(os.path.join(f_out, f"{CFG.model_name}_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    plot_hyperparameter_search_results(search, f_out)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_excel(os.path.join(f_out, f"{CFG.model_name}_report.xlsx"))

    pred_df = pd.DataFrame({
        "True_Label": le.inverse_transform(y_test),
        "Predicted_Label": le.inverse_transform(y_pred),
        "Prediction_Correct": (y_test == y_pred)
    })
    for i, cls in enumerate(class_names):
        pred_df[f"Prob_{cls}"] = y_proba[:, i]
    pred_df.to_csv(os.path.join(f_out, f"{CFG.model_name}_predictions.csv"), index=False)
    pred_df.to_excel(os.path.join(f_out, f"{CFG.model_name}_predictions.xlsx"), index=False)

    plot_model_evaluation(model, X_test, y_test, class_names, f_out)
    plot_learning_curve(model.get_params(), X_train, y_train, f_out, n_classes=len(class_names))

    shap_dir = shap_analysis_single(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        fold_id=fold_id,
        fold_dir_path=f_out
    )

    fold_metrics = {
        "Model": CFG.model_name,
        "Fold": fold_id,
        "Seed": seed,
        "Test_Size": CFG.test_size,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "F1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Model_Path": model_path,
        "SHAP_Dir": shap_dir,
        "Best_Params": json.dumps(best_params, ensure_ascii=False),
    }
    pd.DataFrame([fold_metrics]).to_excel(os.path.join(f_out, "fold_metrics.xlsx"), index=False)

    return fold_metrics


def run_target_pipeline(target_column: str):
    print("\n" + "=" * 60)
    print(f"[{CFG.model_name}] 开始分析目标变量: {target_column}")
    print("=" * 60)

    base_out = target_out_dir(target_column)

    X, y, le, feature_names, class_names = load_data(target_column)

    all_fold_metrics = []
    for fold_id, seed in enumerate(CFG.outer_seeds, start=1):
        print(f"\n[{CFG.model_name}] ===== Fold {fold_id} / Seed {seed} =====")
        fm = run_single_fold(
            X=X, y=y, le=le,
            feature_names=feature_names, class_names=class_names,
            base_target_out=base_out,
            fold_id=fold_id, seed=seed
        )
        fm["Target"] = target_column
        all_fold_metrics.append(fm)

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_excel(os.path.join(base_out, f"{CFG.model_name}_5fold_metrics.xlsx"), index=False)

    aggregate_shap_results(base_out)

    summary = {
        "Model": CFG.model_name,
        "Target": target_column,
        "Accuracy_mean": float(metrics_df["Accuracy"].mean()),
        "Accuracy_sd": float(metrics_df["Accuracy"].std(ddof=1)),
        "F1_weighted_mean": float(metrics_df["F1_weighted"].mean()),
        "F1_weighted_sd": float(metrics_df["F1_weighted"].std(ddof=1)),
    }
    pd.DataFrame([summary]).to_excel(os.path.join(base_out, f"{CFG.model_name}_target_summary.xlsx"), index=False)

    print(f"\n[{CFG.model_name}] 目标变量 {target_column} 全流程完成。输出目录：{base_out}")
    return summary


def main():
    targets = [
        "River Habitat Types",
        "River Formation Classes",
        "Non-native Species Invasion Gradient"
    ]

    all_summaries = []
    for t in targets:
        all_summaries.append(run_target_pipeline(t))

    final_df = pd.DataFrame(all_summaries)
    final_path = os.path.join(CFG.base_output_root, f"{CFG.model_name}_all_targets_summary.xlsx")
    final_df.to_excel(final_path, index=False)

    print("\n" + "=" * 60)
    print(f"[{CFG.model_name}] 所有目标变量完成！综合结果已保存：{final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
