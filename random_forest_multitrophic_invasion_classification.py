
# ============================================================
# RandomForest Full Pipeline (data-only version)
# - Outer: 5 repeated stratified splits (80/20) with seeds 42..46
# - Inner: RandomizedSearchCV + 5-fold StratifiedKFold tuning
# - Per-fold: train/eval + SAVE SHAP RAW ARTIFACTS (for aggregation)
# - FINAL: keep ONLY data outputs, no figure/HTML rendering
# - Aggregation: merge 5 folds shap data -> aggregated_shap_results/*
# ============================================================

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize

import shap


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


DATA_EXCEL_PATH = r"E:\桌面\水生态项目文件\深圳源清项目\2025.5文章数据整理\RobustScaler标准化结果.xlsx"
DATA_SHEET_NAME = "36D_Feature_Matrix"

BASE_OUTPUT_ROOT = r"E:\桌面\水生态项目文件\深圳源清项目\2025.5文章数据整理\1220分组预测结果可视化"

MODEL_NAME = "RandomForest"

OUTER_SEEDS = [42, 43, 44, 45, 46]
OUTER_TEST_SIZE = 0.2

INNER_CV_SPLITS = 5
INNER_CV_RANDOM_STATE = 42

MAX_HTML_INDIVIDUAL_SAMPLES = 300
HTML_SAMPLES_PER_PAGE = 20

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


def safe_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def get_target_output_dir(target_column: str) -> str:
    target_dir = target_column.replace(" ", "_")
    return ensure_dir(os.path.join(BASE_OUTPUT_ROOT, target_dir, MODEL_NAME))


def fold_dir(base_target_out: str, fold_id: int) -> str:
    return ensure_dir(os.path.join(base_target_out, f"fold_{fold_id}"))


def aggregated_dir(base_target_out: str) -> str:
    return ensure_dir(os.path.join(base_target_out, "aggregated_shap_results"))


def rf_shap_root_dir(base_target_out: str) -> str:
    return ensure_dir(os.path.join(base_target_out, f"{MODEL_NAME}_shap"))


def rf_shap_fold_artifact_dir(base_target_out: str, fold_id: int) -> str:
    return ensure_dir(os.path.join(rf_shap_root_dir(base_target_out), "fold_artifacts", f"fold_{fold_id}"))


def infer_group_from_feature(feature: str) -> str:
    for group, prefixes in TROPHIC_GROUPS.items():
        for p in prefixes:
            if feature.startswith(p):
                return group
    return "Other"


def load_data(target_column: str):
    df = pd.read_excel(DATA_EXCEL_PATH, sheet_name=DATA_SHEET_NAME)
    X = df.filter(regex="_st$").astype(np.float32)
    y = df[target_column]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    feature_names = X.columns.tolist()
    class_names = le.classes_
    return X, y_encoded, le, feature_names, class_names


def tune_hyperparameters(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
        "bootstrap": [True, False]
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    rs = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=INNER_CV_RANDOM_STATE),
        scoring=scoring,
        refit="accuracy",
        n_jobs=1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )

    print("\n[RF] 开始 RandomizedSearchCV 超参数调优 ...")
    rs.fit(X_train, y_train)
    print("[RF] 超参数调优完成。best_score =", rs.best_score_)
    return rs.best_estimator_, rs


def plot_hyperparameter_search_results(rs, model_name: str, output_dir: str):
    hp_dir = ensure_dir(os.path.join(output_dir, f"{model_name}_hyperparameters"))

    results = pd.DataFrame(rs.cv_results_)
    results.to_excel(os.path.join(hp_dir, "hyperparameter_search_results.xlsx"), index=False)
    results.to_csv(os.path.join(hp_dir, "hyperparameter_search_results.csv"), index=False)

    params_to_plot = [c for c in results.columns if c.startswith("param_")]
    metrics = ["mean_test_accuracy", "mean_test_precision", "mean_test_recall", "mean_test_f1"]

    results_clean = results.dropna(subset=["mean_test_accuracy"]).copy()

    for param in params_to_plot:
        try:
            export_cols = [param] + [m for m in metrics if m in results_clean.columns]
            param_df = results_clean[export_cols].copy()
            param_name = param.replace("param_", "")
            param_df.to_excel(os.path.join(hp_dir, f"param_{param_name}_plot_data.xlsx"), index=False)
            param_df.to_csv(os.path.join(hp_dir, f"param_{param_name}_plot_data.csv"), index=False)
        except Exception as e:
            print(f"[RF] 导出参数 {param} 绘图数据失败: {str(e)}")
            continue

    try:
        Xp = results_clean[params_to_plot].copy().fillna(0)
        y_score = results_clean["mean_test_accuracy"].copy()

        for col in Xp.columns:
            if Xp[col].dtype == "object":
                Xp[col] = Xp[col].astype(str).astype("category").cat.codes

        rfr = RandomForestRegressor(n_estimators=200, random_state=42)
        rfr.fit(Xp, y_score)

        importance = pd.DataFrame({
            "Parameter": [p.replace("param_", "") for p in params_to_plot],
            "Importance": rfr.feature_importances_
        }).sort_values("Importance", ascending=False)

        importance.to_excel(os.path.join(hp_dir, "hyperparameter_importance.xlsx"), index=False)
        importance.to_csv(os.path.join(hp_dir, "hyperparameter_importance.csv"), index=False)

    except Exception as e:
        print(f"[RF] 参数重要性分析失败: {str(e)}")

    print(f"[RF] 超参数搜索结果数据已保存至: {hp_dir}")


def plot_model_evaluation(model, X_test, y_test, class_names, output_dir):
    eval_dir = ensure_dir(os.path.join(output_dir, f"{MODEL_NAME}_evaluation"))

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

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    if y_test_bin.ndim == 1:
        y_test_bin = y_test_bin.reshape(-1, 1)
    n_classes = y_test_bin.shape[1]

    roc_data_all = []
    roc_auc_summary = []

    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        class_df = pd.DataFrame({
            "Class": [class_names[i]] * len(fpr),
            "FPR": fpr,
            "TPR": tpr,
            "Threshold": thresholds,
            "AUC": [roc_auc] * len(fpr)
        })
        roc_data_all.append(class_df)
        roc_auc_summary.append({
            "Class": class_names[i],
            "AUC": roc_auc
        })

    roc_data_df = pd.concat(roc_data_all, ignore_index=True)
    roc_data_df.to_excel(os.path.join(eval_dir, "roc_curve_raw_data.xlsx"), index=False)
    roc_data_df.to_csv(os.path.join(eval_dir, "roc_curve_raw_data.csv"), index=False)

    roc_auc_df = pd.DataFrame(roc_auc_summary)
    roc_auc_df.to_excel(os.path.join(eval_dir, "roc_auc_summary.xlsx"), index=False)
    roc_auc_df.to_csv(os.path.join(eval_dir, "roc_auc_summary.csv"), index=False)

    prob_df = pd.DataFrame({
        "True_Label_Encoded": np.array(y_test).reshape(-1),
        "Predicted_Label_Encoded": np.array(y_pred).reshape(-1)
    })
    for i, cls in enumerate(class_names):
        prob_df[f"Prob_{cls}"] = y_proba[:, i]
    prob_df.to_excel(os.path.join(eval_dir, "prediction_probabilities.xlsx"), index=False)
    prob_df.to_csv(os.path.join(eval_dir, "prediction_probabilities.csv"), index=False)

    return eval_dir


def plot_learning_curve_randomforest(model, X_train, y_train, output_dir):
    curve_dir = ensure_dir(os.path.join(output_dir, f"{MODEL_NAME}_learning_curve"))

    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    train_sizes_frac = np.unique(np.linspace(0.1, 1.0, num=30, endpoint=True))

    all_train_scores = []
    all_test_scores = []

    actual_sizes = [max(1, int(frac * int(0.8 * len(X_train)))) for frac in train_sizes_frac]

    for frac in tqdm(train_sizes_frac, desc="[RF] Learning curve"):
        fold_train_scores = []
        fold_test_scores = []

        for tr_idx, val_idx in splitter.split(X_train, y_train):
            n_samples = max(1, int(frac * len(tr_idx)))
            tr_sub = tr_idx[:n_samples]

            mdl = RandomForestClassifier(**model.get_params())
            mdl.fit(X_train.iloc[tr_sub], y_train[tr_sub])

            fold_train_scores.append(mdl.score(X_train.iloc[tr_sub], y_train[tr_sub]))
            fold_test_scores.append(mdl.score(X_train.iloc[val_idx], y_train[val_idx]))

        all_train_scores.append(fold_train_scores)
        all_test_scores.append(fold_test_scores)

    train_scores = np.array(all_train_scores)
    test_scores = np.array(all_test_scores)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    curve_data = {
        "train_sizes": actual_sizes,
        "train_scores": train_scores.tolist(),
        "test_scores": test_scores.tolist(),
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "test_mean": test_mean.tolist(),
        "test_std": test_std.tolist()
    }
    with open(os.path.join(curve_dir, "learning_curve_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)

    records = []
    for i, size in enumerate(actual_sizes):
        for fold in range(train_scores.shape[1]):
            records.append({
                "train_size": size,
                "fold": fold + 1,
                "train_score": float(train_scores[i, fold]),
                "test_score": float(test_scores[i, fold]),
                "train_mean": float(train_mean[i]),
                "train_std": float(train_std[i]),
                "test_mean": float(test_mean[i]),
                "test_std": float(test_std[i])
            })
    records_df = pd.DataFrame(records)
    records_df.to_csv(os.path.join(curve_dir, "learning_curve_fold_details.csv"), index=False)
    records_df.to_excel(os.path.join(curve_dir, "learning_curve_fold_details.xlsx"), index=False)

    summary_df = pd.DataFrame({
        "train_size": actual_sizes,
        "train_mean": train_mean,
        "train_std": train_std,
        "test_mean": test_mean,
        "test_std": test_std
    })
    summary_df.to_excel(os.path.join(curve_dir, "learning_curve_summary.xlsx"), index=False)
    summary_df.to_csv(os.path.join(curve_dir, "learning_curve_summary.csv"), index=False)

    return curve_dir


def _normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        return [np.array(v) for v in shap_values]

    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 2:
            return [shap_values]
        if shap_values.ndim == 3:
            k = shap_values.shape[2]
            return [shap_values[:, :, i] for i in range(k)]

    return [np.array(shap_values)]


def _normalize_expected_value(expected_value, n_classes):
    if isinstance(expected_value, list):
        ev = np.array(expected_value, dtype=float)
        if ev.size == n_classes:
            return ev
        if ev.size == 1:
            return np.array([float(ev[0])] * n_classes)
        return np.array([float(ev.flat[0])] * n_classes)

    if isinstance(expected_value, (float, np.floating, int)):
        return np.array([float(expected_value)] * n_classes)

    try:
        ev = np.array(expected_value, dtype=float)
        if ev.size == n_classes:
            return ev
        if ev.size == 1:
            return np.array([float(ev[0])] * n_classes)
        return np.array([float(ev.flat[0])] * n_classes)
    except Exception:
        return np.array([0.0] * n_classes)


def _save_fold_shap_artifacts(
    base_out: str,
    fold_id: int,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: list,
    class_names: np.ndarray,
    shap_values_list: list,
    expected_value: np.ndarray
):
    fdir = rf_shap_fold_artifact_dir(base_out, fold_id)

    X_test.to_csv(os.path.join(fdir, "X_test.csv"), index=False)
    X_test.to_excel(os.path.join(fdir, "X_test.xlsx"), index=False)
    np.save(os.path.join(fdir, "y_test.npy"), np.array(y_test))
    np.save(os.path.join(fdir, "y_pred.npy"), np.array(y_pred))
    np.save(os.path.join(fdir, "y_proba.npy"), np.array(y_proba))

    npz_dict = {}
    for i, sv in enumerate(shap_values_list):
        npz_dict[f"class_{i}"] = np.array(sv, dtype=float)
    np.savez_compressed(os.path.join(fdir, "shap_values.npz"), **npz_dict)

    np.save(os.path.join(fdir, "expected_value.npy"), np.array(expected_value, dtype=float))

    meta = {
        "fold": int(fold_id),
        "n_samples": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1]),
        "feature_names": feature_names,
        "class_names": [str(c) for c in class_names],
    }
    with open(os.path.join(fdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    records = []
    usable_classes = min(len(class_names), len(shap_values_list))
    for c_idx in range(usable_classes):
        cls = str(class_names[c_idx])
        sv = shap_values_list[c_idx]
        for i in range(X_test.shape[0]):
            row_x = X_test.iloc[i].values
            for f_idx, feat in enumerate(feature_names):
                records.append({
                    "Fold": fold_id,
                    "Group": infer_group_from_feature(feat),
                    "Class": cls,
                    "Feature": feat,
                    "ShapValue": float(sv[i, f_idx]),
                    "FeatureValue": float(row_x[f_idx]),
                    "SampleIndex": int(i)
                })
    beeswarm_df = pd.DataFrame(records)
    beeswarm_df.to_csv(os.path.join(fdir, "beeswarm_data.csv"), index=False)
    beeswarm_df.to_excel(os.path.join(fdir, "beeswarm_data.xlsx"), index=False)
    beeswarm_df.to_csv(os.path.join(fdir, "dependence_data.csv"), index=False)
    beeswarm_df.to_excel(os.path.join(fdir, "dependence_data.xlsx"), index=False)

    per_class_importance = {}
    for c_idx in range(usable_classes):
        cls = str(class_names[c_idx])
        per_class_importance[cls] = np.mean(np.abs(shap_values_list[c_idx]), axis=0)

    if usable_classes > 1:
        global_importance = np.mean(np.abs(np.stack(shap_values_list[:usable_classes], axis=0)), axis=(0, 1))
    else:
        global_importance = np.mean(np.abs(shap_values_list[0]), axis=0)

    imp_df = pd.DataFrame({"Feature": feature_names})
    for cls, imp in per_class_importance.items():
        imp_df[f"Importance_{cls}"] = imp
    imp_df["Global_Importance"] = global_importance
    imp_df.sort_values("Global_Importance", ascending=False, inplace=True)
    imp_df.to_excel(os.path.join(fdir, "feature_importance.xlsx"), index=False)
    imp_df.to_csv(os.path.join(fdir, "feature_importance.csv"), index=False)

    fold_summary = {
        "fold": int(fold_id),
        "global_importance": safe_to_list(global_importance),
        "per_class_importance": {k: safe_to_list(v) for k, v in per_class_importance.items()},
    }
    with open(os.path.join(fdir, "shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    print(f"[RF][Fold {fold_id}] SHAP fold artifacts saved: {fdir}")
    return fdir


def aggregate_shap_results(base_target_out: str):
    agg_dir = aggregated_dir(base_target_out)

    beeswarm_all = []
    dep_all = []
    importance_all = []

    for fold_id in range(1, len(OUTER_SEEDS) + 1):
        fdir = rf_shap_fold_artifact_dir(base_target_out, fold_id)
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
            importance_all.append(tmp)

    if len(beeswarm_all) == 0 or len(importance_all) == 0:
        print("[RF] aggregate_shap_results: 没有找到 fold shap 输出，跳过聚合。")
        return None

    combined_beeswarm = pd.concat(beeswarm_all, ignore_index=True)
    combined_dependence = pd.concat(dep_all, ignore_index=True) if len(dep_all) else combined_beeswarm.copy()

    combined_beeswarm.to_csv(os.path.join(agg_dir, "combined_beeswarm_data.csv"), index=False)
    combined_beeswarm.to_excel(os.path.join(agg_dir, "combined_beeswarm_data.xlsx"), index=False)
    combined_dependence.to_csv(os.path.join(agg_dir, "combined_dependence_data.csv"), index=False)
    combined_dependence.to_excel(os.path.join(agg_dir, "combined_dependence_data.xlsx"), index=False)

    imp_all = pd.concat(importance_all, ignore_index=True)

    global_stats = (
        imp_all[["Feature", "Global_Importance"]]
        .groupby("Feature")["Global_Importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    global_stats.rename(columns={"mean": "Global_Mean", "std": "Global_SD"}, inplace=True)

    class_cols = [c for c in imp_all.columns if c.startswith("Importance_")]
    merged = global_stats
    for c in class_cols:
        st = (
            imp_all[["Feature", c]]
            .groupby("Feature")[c]
            .agg(["mean", "std"])
            .reset_index()
        )
        st.rename(columns={"mean": f"{c}_Mean", "std": f"{c}_SD"}, inplace=True)
        merged = merged.merge(st, on="Feature", how="left")

    merged.to_excel(os.path.join(agg_dir, "aggregated_feature_importance.xlsx"), index=False)
    merged.to_csv(os.path.join(agg_dir, "aggregated_feature_importance.csv"), index=False)

    summary_json = {
        "model": MODEL_NAME,
        "n_folds": len(OUTER_SEEDS),
        "top20_global": merged.head(20).to_dict(orient="records")
    }
    with open(os.path.join(agg_dir, "aggregated_shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[RF] 聚合 SHAP 结果保存至: {agg_dir}")
    return agg_dir


def build_final_shap_outputs(base_target_out: str, feature_names: list, class_names: np.ndarray):
    out_dir = rf_shap_root_dir(base_target_out)
    ensure_dir(os.path.join(out_dir, "fold_artifacts"))

    X_all = []
    y_test_all = []
    y_pred_all = []
    y_proba_all = []
    shap_all_per_class = None
    expected_vals = []

    usable_class_count = len(class_names)

    for fold_id in range(1, len(OUTER_SEEDS) + 1):
        fdir = rf_shap_fold_artifact_dir(base_target_out, fold_id)
        x_path = os.path.join(fdir, "X_test.csv")
        sv_path = os.path.join(fdir, "shap_values.npz")
        ev_path = os.path.join(fdir, "expected_value.npy")
        yt_path = os.path.join(fdir, "y_test.npy")
        yp_path = os.path.join(fdir, "y_pred.npy")
        yproba_path = os.path.join(fdir, "y_proba.npy")

        if not (os.path.exists(x_path) and os.path.exists(sv_path) and os.path.exists(ev_path)):
            print(f"[RF] Fold {fold_id} shap artifacts missing, skip.")
            continue

        Xf = pd.read_csv(x_path)
        X_all.append(Xf)

        if os.path.exists(yt_path):
            y_test_all.append(np.load(yt_path))
        if os.path.exists(yp_path):
            y_pred_all.append(np.load(yp_path))
        if os.path.exists(yproba_path):
            y_proba_all.append(np.load(yproba_path))

        sv_npz = np.load(sv_path)
        keys = sorted([k for k in sv_npz.files if k.startswith("class_")], key=lambda z: int(z.split("_")[1]))
        sv_list = [sv_npz[k] for k in keys]
        fold_k = min(usable_class_count, len(sv_list))

        if shap_all_per_class is None:
            shap_all_per_class = [[] for _ in range(fold_k)]

        for c in range(fold_k):
            shap_all_per_class[c].append(sv_list[c])

        ev = np.load(ev_path)
        ev = _normalize_expected_value(ev, fold_k)
        expected_vals.append(ev)

    if len(X_all) == 0 or shap_all_per_class is None:
        print("[RF] build_final_shap_outputs: 没有足够的 fold shap artifacts，跳过最终数据输出。")
        return None

    X_concat = pd.concat(X_all, ignore_index=True)

    shap_concat_list = []
    for c in range(len(shap_all_per_class)):
        shap_concat_list.append(np.vstack(shap_all_per_class[c]))

    expected_val = np.mean(np.vstack(expected_vals), axis=0)

    per_class_importance = {}
    for c_idx in range(len(shap_concat_list)):
        cls = str(class_names[c_idx]) if c_idx < len(class_names) else f"class_{c_idx}"
        per_class_importance[cls] = np.mean(np.abs(shap_concat_list[c_idx]), axis=0)

    if len(shap_concat_list) > 1:
        global_importance = np.mean(np.abs(np.stack(shap_concat_list, axis=0)), axis=(0, 1))
    else:
        global_importance = np.mean(np.abs(shap_concat_list[0]), axis=0)

    importance_df = pd.DataFrame({"Feature": feature_names})
    for cls, imp in per_class_importance.items():
        importance_df[f"Importance_{cls}"] = imp
    importance_df["Global_Importance"] = global_importance
    importance_df.sort_values("Global_Importance", ascending=False, inplace=True)
    importance_df.to_excel(os.path.join(out_dir, "feature_importance.xlsx"), index=False)
    importance_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    records = []
    for c_idx in range(len(shap_concat_list)):
        cls = str(class_names[c_idx]) if c_idx < len(class_names) else f"class_{c_idx}"
        sv = shap_concat_list[c_idx]
        for i in range(X_concat.shape[0]):
            row_x = X_concat.iloc[i].values
            for f_idx, feat in enumerate(feature_names):
                records.append({
                    "Class": cls,
                    "Feature": feat,
                    "Group": infer_group_from_feature(feat),
                    "ShapValue": float(sv[i, f_idx]),
                    "FeatureValue": float(row_x[f_idx]),
                    "SampleIndex": int(i)
                })
    agg_shap_df = pd.DataFrame(records)
    agg_shap_df.to_csv(os.path.join(out_dir, "aggregated_beeswarm_data.csv"), index=False)
    agg_shap_df.to_excel(os.path.join(out_dir, "aggregated_beeswarm_data.xlsx"), index=False)
    agg_shap_df.to_csv(os.path.join(out_dir, "aggregated_dependence_data.csv"), index=False)
    agg_shap_df.to_excel(os.path.join(out_dir, "aggregated_dependence_data.xlsx"), index=False)

    output_summary = {
        "n_samples": int(X_concat.shape[0]),
        "n_features": int(X_concat.shape[1]),
        "feature_names": feature_names,
        "class_names": [str(c) for c in class_names],
        "expected_value": safe_to_list(expected_val),
        "global_importance": safe_to_list(global_importance),
        "per_class_importance": {k: safe_to_list(v) for k, v in per_class_importance.items()}
    }
    with open(os.path.join(out_dir, "aggregated_shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    if len(y_test_all) > 0:
        y_test_concat = np.concatenate(y_test_all)
    else:
        y_test_concat = np.array([])
    if len(y_pred_all) > 0:
        y_pred_concat = np.concatenate(y_pred_all)
    else:
        y_pred_concat = np.array([])
    if len(y_proba_all) > 0:
        y_proba_concat = np.concatenate(y_proba_all)
    else:
        y_proba_concat = np.empty((0, len(class_names)))

    pred_df = pd.DataFrame({
        "True_Label_Encoded": y_test_concat,
        "Predicted_Label_Encoded": y_pred_concat
    })
    if y_proba_concat.size > 0:
        for i, cls in enumerate(class_names):
            if i < y_proba_concat.shape[1]:
                pred_df[f"Prob_{cls}"] = y_proba_concat[:, i]
    pred_df.to_csv(os.path.join(out_dir, "aggregated_prediction_probabilities.csv"), index=False)
    pred_df.to_excel(os.path.join(out_dir, "aggregated_prediction_probabilities.xlsx"), index=False)

    print(f"[RF] Final SHAP data outputs saved to: {out_dir}")
    return out_dir


def run_target_pipeline(target_column: str):
    print(f"\n{'=' * 60}")
    print(f"[RF] 开始分析目标变量: {target_column}")
    print(f"{'=' * 60}")

    base_out = get_target_output_dir(target_column)
    _ = rf_shap_root_dir(base_out)

    X, y, le, feature_names, class_names = load_data(target_column)

    all_fold_metrics = []

    for fold_id, seed in enumerate(OUTER_SEEDS, start=1):
        print(f"\n[RF] ===== Fold {fold_id} / Seed {seed} =====")
        f_out = fold_dir(base_out, fold_id)

        idx_all = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=OUTER_TEST_SIZE,
            stratify=y,
            random_state=seed
        )

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        model, search = tune_hyperparameters(X_train, y_train)

        model_path = os.path.join(f_out, f"{MODEL_NAME}.pkl")
        joblib.dump(model, model_path)

        best_params = search.best_params_
        with open(os.path.join(f_out, f"{MODEL_NAME}_best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

        plot_hyperparameter_search_results(search, MODEL_NAME, f_out)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_excel(os.path.join(f_out, f"{MODEL_NAME}_report.xlsx"))

        pred_df = pd.DataFrame({
            "True_Label": le.inverse_transform(y_test),
            "Predicted_Label": le.inverse_transform(y_pred),
            "Prediction_Correct": (y_test == y_pred)
        })
        for i, cls in enumerate(le.classes_):
            pred_df[f"Prob_{cls}"] = y_proba[:, i]
        pred_df.to_csv(os.path.join(f_out, f"{MODEL_NAME}_predictions.csv"), index=False)
        pred_df.to_excel(os.path.join(f_out, f"{MODEL_NAME}_predictions.xlsx"), index=False)

        plot_learning_curve_randomforest(model, X_train, y_train, f_out)
        plot_model_evaluation(model, X_test, y_test, class_names, f_out)

        print(f"[RF][Fold {fold_id}] 计算 SHAP（仅保存 fold 原始 artifacts，最终只出数据）...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_values_list = _normalize_shap_values(shap_values)

        usable_classes = min(len(class_names), len(shap_values_list))
        shap_values_list = shap_values_list[:usable_classes]

        expected_value = _normalize_expected_value(explainer.expected_value, usable_classes)

        fold_art_dir = _save_fold_shap_artifacts(
            base_out=base_out,
            fold_id=fold_id,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            feature_names=feature_names,
            class_names=class_names,
            shap_values_list=shap_values_list,
            expected_value=expected_value
        )

        fold_metrics = {
            "Model": MODEL_NAME,
            "Target": target_column,
            "Fold": fold_id,
            "Seed": seed,
            "Test_Size": OUTER_TEST_SIZE,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Model_Path": model_path,
            "SHAP_Fold_Artifacts": fold_art_dir,
            "Best_Params": json.dumps(best_params, ensure_ascii=False)
        }
        all_fold_metrics.append(fold_metrics)
        pd.DataFrame([fold_metrics]).to_excel(os.path.join(f_out, "fold_metrics.xlsx"), index=False)

        print(f"[RF][Fold {fold_id}] 完成。Fold 输出：{f_out}")

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    fold_metrics_df.to_excel(os.path.join(base_out, f"{MODEL_NAME}_5fold_metrics.xlsx"), index=False)

    aggregate_shap_results(base_out)
    build_final_shap_outputs(base_out, feature_names, class_names)

    summary = {
        "Model": MODEL_NAME,
        "Target": target_column,
        "Accuracy_mean": float(fold_metrics_df["Accuracy"].mean()),
        "Accuracy_sd": float(fold_metrics_df["Accuracy"].std(ddof=1)),
        "F1_weighted_mean": float(fold_metrics_df["F1_weighted"].mean()),
        "F1_weighted_sd": float(fold_metrics_df["F1_weighted"].std(ddof=1))
    }
    pd.DataFrame([summary]).to_excel(os.path.join(base_out, f"{MODEL_NAME}_target_summary.xlsx"), index=False)

    print(f"\n[RF] 目标变量 {target_column} 全流程完成。输出目录：{base_out}")
    print(f"[RF] 最终 SHAP 数据输出目录：{rf_shap_root_dir(base_out)}")
    return summary


def main():
    target_columns = [
        "River Habitat Types",
        "River Formation Classes",
        "Non-native Species Invasion Gradient"
    ]

    all_summaries = []
    for target in target_columns:
        s = run_target_pipeline(target)
        all_summaries.append(s)

    final_df = pd.DataFrame(all_summaries)
    final_path = os.path.join(BASE_OUTPUT_ROOT, f"{MODEL_NAME}_all_targets_summary.xlsx")
    final_df.to_excel(final_path, index=False)

    print("\n" + "=" * 60)
    print(f"[RF] 所有目标变量完成！综合结果已保存：{final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
