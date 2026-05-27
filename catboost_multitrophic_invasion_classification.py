# ============================================================
# CatBoost Full Pipeline (XGBoost-aligned, full version)
# - Outer: 5 repeated stratified splits (80/20) with seeds 42..46
# - Inner: RandomizedSearchCV + 5-fold StratifiedKFold tuning
# - Outputs: same fold logic, directory structure, file checklist
# - SHAP: TreeExplainer, grouped by trophic prefixes, save data + figures + HTML
# - Aggregation: merge 5 folds shap data, compute mean±sd importance
# ============================================================

import os
import json
import joblib
import warnings
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV,
    StratifiedShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize

import shap


# ============================================================
# 0) Global settings (match your code style)
# ============================================================
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


# ============================================================
# 1) Paths & constants (aligned)
# ============================================================
DATA_EXCEL_PATH = r"E:\桌面\水生态项目文件\深圳源清项目\2025.5文章数据整理\RobustScaler标准化结果.xlsx"
DATA_SHEET_NAME = "36D_Feature_Matrix"

BASE_OUTPUT_ROOT = r"E:\桌面\水生态项目文件\深圳源清项目\2025.5文章数据整理\1220分组预测结果可视化"

MODEL_NAME = "CatBoost"

OUTER_SEEDS = [42, 43, 44, 45, 46]
OUTER_TEST_SIZE = 0.2

INNER_CV_SPLITS = 5
INNER_CV_RANDOM_STATE = 42

# Multi-trophic prefixes
TROPHIC_GROUPS = {
    "Phytoplankton": ["P_"],
    "PeriphyticAlgae": ["PA_"],
    "BenthicMacroinvertebrates": ["BM_"],
    "Zooplankton": ["Z_"],
    "AquaticPlants": ["AP_"],
    "Fish": ["F_"]
}


# ============================================================
# 2) Utilities
# ============================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def get_target_output_dir(target_column: str) -> str:
    # XGBoost-style: BASE_ROOT / target / model
    target_dir = target_column.replace(" ", "_")
    return ensure_dir(os.path.join(BASE_OUTPUT_ROOT, target_dir, MODEL_NAME))


def fold_dir(base_target_out: str, fold_id: int) -> str:
    return ensure_dir(os.path.join(base_target_out, f"fold_{fold_id}"))


def aggregated_dir(base_target_out: str) -> str:
    return ensure_dir(os.path.join(base_target_out, "aggregated_shap_results"))


def infer_group_from_feature(feature: str) -> str:
    for group, prefixes in TROPHIC_GROUPS.items():
        for p in prefixes:
            if feature.startswith(p):
                return group
    return "Other"


# ============================================================
# 3) Data loading
# ============================================================
def load_data(target_column: str):
    df = pd.read_excel(DATA_EXCEL_PATH, sheet_name=DATA_SHEET_NAME)
    X = df.filter(regex="_st$").astype(np.float32)
    y = df[target_column]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    feature_names = X.columns.tolist()
    class_names = le.classes_
    return X, y_encoded, le, feature_names, class_names


# ============================================================
# 4) Hyperparameter tuning (STRICTLY from your CatBoost source)
# ============================================================
def tune_hyperparameters(X_train, y_train):
    # EXACT ranges from your provided CatBoost code
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "depth": [3, 5, 7, 10],
        "l2_leaf_reg": [1, 3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bylevel": [0.6, 0.8, 1.0],
        "iterations": [50, 100, 200, 300],
        "min_child_samples": [1, 5, 10],
        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
        "random_strength": [0.1, 1, 10],
    }

    base_model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        verbose=0,
        task_type="CPU"
    )

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

    print("\n[CatBoost] 开始 RandomizedSearchCV 超参数调优 ...")
    rs.fit(X_train, y_train)
    print("[CatBoost] 超参数调优完成。best_score =", rs.best_score_)
    return rs.best_estimator_, rs


# ============================================================
# 5) Hyperparameter search visualization (same as your style)
# ============================================================
def plot_hyperparameter_search_results(rs, model_name: str, output_dir: str):
    hp_dir = ensure_dir(os.path.join(output_dir, f"{model_name}_hyperparameters"))

    results = pd.DataFrame(rs.cv_results_)
    results.to_excel(os.path.join(hp_dir, "hyperparameter_search_results.xlsx"), index=False)

    params_to_plot = [c for c in results.columns if c.startswith("param_")]
    metrics = ["mean_test_accuracy", "mean_test_precision", "mean_test_recall", "mean_test_f1"]

    # Clean NaN rows (you already did this in your CatBoost code)
    results_clean = results.dropna(subset=["mean_test_accuracy"]).copy()

    for param in params_to_plot:
        try:
            plt.figure(figsize=(12, 6))
            is_numeric = pd.api.types.is_numeric_dtype(results_clean[param])

            if is_numeric:
                for metric in metrics:
                    plt.scatter(
                        results_clean[param],
                        results_clean[metric],
                        alpha=0.5,
                        label=metric.replace("mean_test_", "").capitalize()
                    )
                plt.xlabel(param.replace("param_", ""))
                plt.ylabel("Score")
                plt.title(f"{model_name} - Hyperparameter: {param.replace('param_', '')}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(hp_dir, f"param_{param.replace('param_', '')}_scatter.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close()
            else:
                plt.figure(figsize=(12, 6))
                sns.boxplot(
                    data=results_clean,
                    x=param,
                    y="mean_test_accuracy",
                    palette="viridis"
                )
                plt.title(f"{model_name} - Accuracy by {param.replace('param_', '')}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(hp_dir, f"param_{param.replace('param_', '')}_boxplot.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close()
        except Exception as e:
            print(f"[CatBoost] 绘制参数 {param} 可视化失败: {str(e)}")
            plt.close()
            continue

    # Hyperparameter importance (same approach as your code)
    try:
        from sklearn.ensemble import RandomForestRegressor

        Xp = results_clean[params_to_plot].copy().fillna(0)
        y_score = results_clean["mean_test_accuracy"].copy()

        for col in Xp.columns:
            if Xp[col].dtype == "object":
                Xp[col] = Xp[col].astype(str).astype("category").cat.codes

        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(Xp, y_score)

        importance = pd.DataFrame({
            "Parameter": [p.replace("param_", "") for p in params_to_plot],
            "Importance": rfr.feature_importances_
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance, x="Importance", y="Parameter", palette="viridis")
        plt.title(f"{model_name} - Hyperparameter Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(hp_dir, "hyperparameter_importance.png"), dpi=300, bbox_inches="tight")
        plt.close()

        importance.to_excel(os.path.join(hp_dir, "hyperparameter_importance.xlsx"), index=False)

    except Exception as e:
        print(f"[CatBoost] 参数重要性分析失败: {str(e)}")
        plt.close()

    print(f"[CatBoost] 超参数搜索结果可视化已保存至: {hp_dir}")


# ============================================================
# 6) Evaluation plots (confusion matrix + ROC)
# ============================================================
def plot_model_evaluation(model, X_test, y_test, class_names, output_dir):
    eval_dir = ensure_dir(os.path.join(output_dir, f"{MODEL_NAME}_evaluation"))

    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # CatBoost may return shape (n,1); flatten
    y_pred = np.array(y_pred).reshape(-1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 28}, cbar=False
    )
    plt.title(f"{MODEL_NAME} - Confusion Matrix", fontsize=26)
    plt.xlabel("Predicted Label", fontsize=22)
    plt.ylabel("True Label", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ROC (OvR)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(10, 8))
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])

    # ROC (OvR) + 保存 ROC 原始数据
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]



    return eval_dir


# ============================================================
# 7) Learning curve (XGBoost-aligned: fixed validation via StratifiedShuffleSplit)
# ============================================================
def plot_learning_curve(model, X_train, y_train, output_dir):
    curve_dir = ensure_dir(os.path.join(output_dir, f"{MODEL_NAME}_learning_curve"))

    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    train_sizes_frac = np.unique(np.linspace(0.1, 1.0, num=30, endpoint=True))

    all_train_scores = []
    all_test_scores = []
    actual_sizes = [max(1, int(frac * int(0.8 * len(X_train)))) for frac in train_sizes_frac]

    for frac in tqdm(train_sizes_frac, desc="[CatBoost] Learning curve"):
        fold_train_scores = []
        fold_test_scores = []

        for tr_idx, val_idx in splitter.split(X_train, y_train):
            n_samples = max(1, int(frac * len(tr_idx)))
            tr_sub = tr_idx[:n_samples]

            # Clone-like rebuild (CatBoost doesn't always behave perfectly with sklearn.clone)
            mdl = CatBoostClassifier(**model.get_params())
            mdl.set_params(verbose=0)
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

    plt.figure(figsize=(10, 6))
    plt.plot(actual_sizes, train_mean, marker="o", label="Training score")
    plt.plot(actual_sizes, test_mean, marker="o", label="Cross-validation score")
    plt.fill_between(actual_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(actual_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
    plt.title(f"{MODEL_NAME} - Learning Curve", fontsize=26)
    plt.xlabel("Number of training samples", fontsize=22)
    plt.ylabel("Accuracy", fontsize=22)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(curve_dir, "learning_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    curve_data = {
        "train_sizes": actual_sizes,
        "train_scores": safe_to_list(train_scores),
        "test_scores": safe_to_list(test_scores),
        "train_mean": safe_to_list(train_mean),
        "train_std": safe_to_list(train_std),
        "test_mean": safe_to_list(test_mean),
        "test_std": safe_to_list(test_std)
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
                "test_score": float(test_scores[i, fold])
            })
    pd.DataFrame(records).to_csv(os.path.join(curve_dir, "learning_curve_fold_details.csv"), index=False)

    return curve_dir


# ============================================================
# 8) SHAP (XGBoost-aligned data outputs + HTML)
# ============================================================
def shap_analysis_single(model, X_test, y_test, feature_names, class_names, fold_id: int, output_dir: str):
    """
    修复：使用 CatBoost 原生 SHAP，避免 Windows 下 shap.TreeExplainer 崩溃 (0xC0000005)
    """
    shap_dir = ensure_dir(os.path.join(output_dir, f"{MODEL_NAME}_shap"))

    try:
        pool = Pool(X_test, y_test)
        shap_raw = model.get_feature_importance(data=pool, type="ShapValues")
    except Exception as e:
        raise RuntimeError(f"[CatBoost][Fold {fold_id}] SHAP 计算失败: {str(e)}")

    shap_raw = np.array(shap_raw)

    n_samples = X_test.shape[0]
    n_features = len(feature_names)
    n_classes = len(class_names)

    per_class_shap = []

    if shap_raw.ndim == 3:
        for c in range(shap_raw.shape[1]):
            per_class_shap.append(shap_raw[:, c, :n_features])
        expected_value = shap_raw[:, :, -1].mean(axis=0)

    elif shap_raw.ndim == 2:
        per_class_shap = [shap_raw[:, :n_features]]
        expected_value = np.array([shap_raw[:, -1].mean()])
        class_names = class_names[:1]

    else:
        raise ValueError(f"不支持的 SHAP 输出维度: {shap_raw.shape}")

    # ===== 保存数据 =====
    beeswarm_records = []
    for c_idx, cls in enumerate(class_names):
        sv = per_class_shap[c_idx]
        for i in range(n_samples):
            for f_idx, feat in enumerate(feature_names):
                beeswarm_records.append({
                    "Fold": fold_id,
                    "Group": infer_group_from_feature(feat),
                    "Class": str(cls),
                    "Feature": feat,
                    "ShapValue": float(sv[i, f_idx]),
                    "FeatureValue": float(X_test.iloc[i, f_idx]),
                    "SampleIndex": int(i)
                })

    beeswarm_df = pd.DataFrame(beeswarm_records)
    beeswarm_df.to_csv(os.path.join(shap_dir, "beeswarm_data.csv"), index=False)
    beeswarm_df.to_csv(os.path.join(shap_dir, "dependence_data.csv"), index=False)

    # ===== 特征重要性 =====
    per_class_importance = {}
    for c_idx, cls in enumerate(class_names):
        per_class_importance[str(cls)] = np.mean(np.abs(per_class_shap[c_idx]), axis=0)

    if len(per_class_shap) > 1:
        global_importance = np.mean(np.abs(np.stack(per_class_shap, axis=0)), axis=(0, 1))
    else:
        global_importance = np.mean(np.abs(per_class_shap[0]), axis=0)

    importance_df = pd.DataFrame({"Feature": feature_names})
    for cls, imp in per_class_importance.items():
        importance_df[f"Importance_{cls}"] = imp
    importance_df["Global_Importance"] = global_importance
    importance_df.sort_values("Global_Importance", ascending=False, inplace=True)
    importance_df.to_excel(os.path.join(shap_dir, "feature_importance.xlsx"), index=False)

    summary_data = {
        "fold": fold_id,
        "model": MODEL_NAME,
        "n_samples_test": int(n_samples),
        "feature_names": feature_names,
        "class_names": [str(c) for c in class_names],
        "expected_value": safe_to_list(np.array(expected_value)),
        "global_importance": safe_to_list(global_importance),
        "per_class_importance": {k: safe_to_list(v) for k, v in per_class_importance.items()}
    }

    with open(os.path.join(shap_dir, "shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[CatBoost][Fold {fold_id}] SHAP 结果保存至: {shap_dir}")
    return shap_dir


# ============================================================
# 9) Aggregate SHAP results across folds
# ============================================================
def aggregate_shap_results(base_target_out: str):
    agg_dir = aggregated_dir(base_target_out)

    beeswarm_all = []
    dep_all = []
    importance_all = []

    for fold_id in range(1, len(OUTER_SEEDS) + 1):
        fdir = os.path.join(base_target_out, f"fold_{fold_id}", f"{MODEL_NAME}_shap")
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
        print("[CatBoost] aggregate_shap_results: 没有找到 fold shap 输出，跳过聚合。")
        return None

    combined_beeswarm = pd.concat(beeswarm_all, ignore_index=True)
    combined_dependence = pd.concat(dep_all, ignore_index=True) if len(dep_all) else combined_beeswarm.copy()

    combined_beeswarm.to_csv(os.path.join(agg_dir, "combined_beeswarm_data.csv"), index=False)
    combined_dependence.to_csv(os.path.join(agg_dir, "combined_dependence_data.csv"), index=False)

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

    summary_json = {
        "model": MODEL_NAME,
        "n_folds": len(OUTER_SEEDS),
        "top20_global": merged.head(20).to_dict(orient="records")
    }
    with open(os.path.join(agg_dir, "aggregated_shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[CatBoost] 聚合 SHAP 结果保存至: {agg_dir}")
    return agg_dir


# ============================================================
# 10) Run one target (full fold loop)
# ============================================================
def run_target_pipeline(target_column: str):
    print(f"\n{'=' * 60}")
    print(f"[CatBoost] 开始分析目标变量: {target_column}")
    print(f"{'=' * 60}")

    base_out = get_target_output_dir(target_column)

    X, y, le, feature_names, class_names = load_data(target_column)

    all_fold_metrics = []

    for fold_id, seed in enumerate(OUTER_SEEDS, start=1):
        print(f"\n[CatBoost] ===== Fold {fold_id} / Seed {seed} =====")
        f_out = fold_dir(base_out, fold_id)

        # Outer stratified split (80/20)
        idx_all = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=OUTER_TEST_SIZE,
            stratify=y,
            random_state=seed
        )

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # 1) Tune
        model, search = tune_hyperparameters(X_train, y_train)

        # 2) Save model & best params
        model_path = os.path.join(f_out, f"{MODEL_NAME}.pkl")
        joblib.dump(model, model_path)

        best_params = search.best_params_
        with open(os.path.join(f_out, f"{MODEL_NAME}_best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

        # 3) Hyperparameter plots & exports
        plot_hyperparameter_search_results(search, MODEL_NAME, f_out)

        # 4) Predict + report + predictions
        y_pred = model.predict(X_test)
        y_pred = np.array(y_pred).reshape(-1)

        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_excel(os.path.join(f_out, f"{MODEL_NAME}_report.xlsx"))

        pred_df = pd.DataFrame({
            "True_Label": le.inverse_transform(np.array(y_test).reshape(-1)),
            "Predicted_Label": le.inverse_transform(np.array(y_pred).reshape(-1)),
            "Prediction_Correct": (np.array(y_test).reshape(-1) == np.array(y_pred).reshape(-1))
        })
        pred_df.to_csv(os.path.join(f_out, f"{MODEL_NAME}_predictions.csv"), index=False)

        # 5) Learning curve
        plot_learning_curve(model, X_train, y_train, f_out)

        # 6) Evaluation plots
        plot_model_evaluation(model, X_test, y_test, class_names, f_out)

        # 7) SHAP
        print(f"[CatBoost][Fold {fold_id}] 开始 SHAP 分析 ...")
        shap_dir = shap_analysis_single(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            class_names=class_names,
            fold_id=fold_id,
            output_dir=f_out
        )

        # 8) Fold metrics
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
            "SHAP_Dir": shap_dir,
            "Best_Params": json.dumps(best_params, ensure_ascii=False)
        }
        all_fold_metrics.append(fold_metrics)
        pd.DataFrame([fold_metrics]).to_excel(os.path.join(f_out, "fold_metrics.xlsx"), index=False)

        print(f"[CatBoost][Fold {fold_id}] 完成。结果保存至: {f_out}")

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    fold_metrics_df.to_excel(os.path.join(base_out, f"{MODEL_NAME}_5fold_metrics.xlsx"), index=False)

    aggregate_shap_results(base_out)

    summary = {
        "Model": MODEL_NAME,
        "Target": target_column,
        "Accuracy_mean": float(fold_metrics_df["Accuracy"].mean()),
        "Accuracy_sd": float(fold_metrics_df["Accuracy"].std(ddof=1)),
        "F1_weighted_mean": float(fold_metrics_df["F1_weighted"].mean()),
        "F1_weighted_sd": float(fold_metrics_df["F1_weighted"].std(ddof=1))
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_excel(os.path.join(base_out, f"{MODEL_NAME}_target_summary.xlsx"), index=False)

    print(f"\n[CatBoost] 目标变量 {target_column} 全流程完成。输出目录：{base_out}")
    return summary_df


# ============================================================
# 11) Main
# ============================================================
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

    final_df = pd.concat(all_summaries, ignore_index=True)
    final_path = os.path.join(BASE_OUTPUT_ROOT, f"{MODEL_NAME}_all_targets_summary.xlsx")
    final_df.to_excel(final_path, index=False)

    print("\n" + "=" * 60)
    print(f"[CatBoost] 所有目标变量完成！综合结果已保存：{final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
