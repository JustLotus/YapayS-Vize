"""Reproducible development, prediction lock and independent evaluation."""
import hashlib
import io
import json
import random
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 31063
SPLIT_SEED = BASE_SEED + 1
SEEDS = [BASE_SEED + 100, BASE_SEED + 101, BASE_SEED + 102]
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
torch.set_num_threads(1)


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, pd.DataFrame):
        value.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def csv(name):
    return pd.read_csv(ROOT / "results" / name)


def plot(name):
    path = ROOT / "figures" / name
    path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=145)
    plt.close("all")


def data():
    path = ROOT / "data" / "bank-additional.csv"
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as outer_zip:
            with zipfile.ZipFile(io.BytesIO(outer_zip.read("bank-additional.zip"))) as archive:
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(archive.read("bank-additional/bank-additional.csv"))
    raw = pd.read_csv(path, sep=";")
    assert raw.shape == (4119, 21) and set(raw.y) == {"yes", "no"}
    return raw.drop(columns=["duration", "y"]), raw.y.map({"yes": 1, "no": 0})


def split():
    x, y = data()
    dev, test = train_test_split(np.arange(len(x)), test_size=.2, stratify=y,
                                 random_state=SPLIT_SEED)
    return x, y, dev, test


def candidates():
    base = dict(optimizer="AdamW", hidden=[32], scaler="Standard", lr=.001,
                wd=.0001, scheduler=True, batch=64, dropout=0., activation="ReLU",
                clip=False)
    changes = [{}, {"optimizer": "Adam"}, {"optimizer": "SGD"}, {"scaler": "MinMax"},
               {"hidden": [64, 32]}, {"scheduler": False}]
    manual = [dict(base, **change, id=f"M{i}") for i, change in enumerate(changes, 1)]
    rng = random.Random(BASE_SEED + 2)
    automatic = []
    for optimizer in ("SGD", "Adam", "AdamW"):
        for _ in range(2):
            automatic.append(dict(
                id=f"R{len(automatic)+1}", optimizer=optimizer,
                hidden=rng.choice([[16], [32], [64], [64, 32], [128, 64]]),
                scaler=rng.choice(["Standard", "MinMax", "Robust"]),
                lr=10 ** rng.uniform(*((-3, -1) if optimizer == "SGD" else (-4, -2))),
                wd=rng.choice([0, 1e-5, 1e-4, 1e-3]), scheduler=True,
                batch=rng.choice([32, 64, 128]), dropout=rng.choice([0., .2]),
                activation=rng.choice(["ReLU", "tanh"]), clip=rng.choice([False, True])))
    return manual, automatic


def prepare():
    x, y, dev, test = split()
    d, target = x.loc[dev], y.loc[dev]
    distribution = target.value_counts().sort_index().rename_axis("class").reset_index(name="count")
    distribution["ratio"] = distribution["count"] / len(dev)
    save("results/class_distribution.csv", distribution)
    save("results/dataset_split.csv", split_table(y, dev, test, reveal=False))
    numeric = d.select_dtypes("number").copy()
    numeric["pdays"] = numeric["pdays"].replace(999, np.nan)
    summary = numeric.describe().T
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    summary["IQR"] = q3 - q1
    flags = numeric.lt(q1 - 1.5 * (q3 - q1)) | numeric.gt(q3 + 1.5 * (q3 - q1))
    summary["IQR_outliers"] = flags.sum()
    save("results/numeric_summary.csv", summary.rename_axis("feature").reset_index())
    save("results/outlier_flags.csv", flags.rename_axis("row_index").reset_index())
    special = pd.DataFrame({"feature": d.columns, "missing": d.isna().sum().values,
                            "unknown": d.eq("unknown").sum().values,
                            "unique": d.nunique().values})
    special["pdays_999"] = np.where(special.feature.eq("pdays"), d.pdays.eq(999).sum(), 0)
    save("results/special_values.csv", special)
    save("results/eda_counts.json", {"duplicate_dev_feature_target_rows":
         int(d.assign(y=target).duplicated().sum()), "development": len(dev), "test": len(test)})
    categories = d.select_dtypes(exclude="number")
    save("results/categorical_frequencies.csv", pd.concat([
        categories[c].value_counts().rename_axis("value").reset_index(name="count").assign(feature=c)
        for c in categories], ignore_index=True))
    save("results/numeric_by_target.csv", numeric.assign(y=target).groupby("y").mean().reset_index())
    distribution.set_index("class")["count"].plot.bar(title="Development class balance", rot=0)
    plot("eda_class_balance.png")
    numeric.hist(figsize=(10, 7))
    plot("eda_histograms.png")
    _, axes = plt.subplots(2, 5, figsize=(11, 5))
    for axis, col in zip(axes.flat, numeric):
        axis.boxplot(numeric[col].dropna())
        axis.set_title(col, fontsize=8)
        axis.set_xticks([])
    plot("eda_boxplots.png")
    corr = numeric.corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(corr)), corr.columns, rotation=65, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar()
    plot("eda_correlations.png")
    categories["job"].value_counts().plot.bar(title="Development job frequencies")
    plot("eda_categories.png")
    manual, automatic = candidates()
    save("configs/manual_candidates.json", manual)
    save("configs/random_candidates.json", automatic)
    save("configs/protocol.json", dict(base_seed=BASE_SEED, split_seed=SPLIT_SEED,
                                     weight_seeds=SEEDS, threshold=.5, max_epochs=150,
                                     patience=15, scheduler_patience=5, bootstrap=1000,
                                     early_stop_fraction=.15, inner_folds=3, outer_folds=3,
                                     unknown_policy="preserve category",
                                     duplicate_policy="preserve original rows, report limitation"))


def split_table(y, dev, test, reveal):
    frames = []
    for name, rows in [("development", dev), ("test", test)]:
        positive = float(y.loc[rows].mean()) if name == "development" or reveal else np.nan
        frames.append(pd.DataFrame(dict(row_index=rows, split=name,
                                        positive_ratio=positive, negative_ratio=1-positive)))
    return pd.concat(frames).sort_values("row_index")


def features(x):
    return x.assign(pdays_999=x.pdays.eq(999).astype(float),
                    pdays=x.pdays.replace(999, np.nan))


def preprocess(x, fit, config):
    x = features(x)
    numeric = list(x.select_dtypes("number"))
    categorical = list(x.select_dtypes(exclude="number"))
    clipcols = [name for name in numeric if name != "pdays_999"]
    low, high = (x.loc[fit, clipcols].quantile(q) for q in (.01, .99)) if config["clip"] else (None, None)
    if low is not None:
        x[clipcols] = x[clipcols].clip(low, high, axis="columns")
    scaler = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}[config["scaler"]]
    pre = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  scaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                      OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical)])
    pre.fit(x.loc[fit])
    return x, pre, low, high


def train(x, y, rows, assess, config, split_seed, weight_seed, run):
    start = time.perf_counter()
    random.seed(weight_seed)
    np.random.seed(weight_seed)
    torch.manual_seed(weight_seed)
    fit, early = train_test_split(rows, test_size=.15, stratify=y.loc[rows],
                                  random_state=split_seed)
    x, pre, low, high = preprocess(x, fit, config)
    transform = lambda indices: torch.tensor(pre.transform(x.loc[indices]), dtype=torch.float32)
    a, b = transform(fit), transform(early)
    ay = torch.tensor(y.loc[fit].to_numpy(), dtype=torch.float32)
    by = torch.tensor(y.loc[early].to_numpy(), dtype=torch.float32)
    widths = [a.shape[1], *config["hidden"]]
    layers = []
    activation = nn.ReLU if config["activation"] == "ReLU" else nn.Tanh
    for left, right in zip(widths, widths[1:]):
        layers.extend([nn.Linear(left, right), activation(), nn.Dropout(config["dropout"])])
    model = nn.Sequential(*layers, nn.Linear(widths[-1], 1))
    kind = config["optimizer"]
    extra = {"momentum": .9} if kind == "SGD" else {}
    opt = getattr(torch.optim, kind)(model.parameters(), lr=config["lr"], weight_decay=config["wd"], **extra)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=5, min_lr=1e-5)
    loss = nn.BCEWithLogitsLoss()
    history = []
    best, best_epoch = float("inf"), 0
    generator = torch.Generator().manual_seed(weight_seed)
    for epoch in range(1, 151):
        model.train()
        total = 0.
        lr = opt.param_groups[0]["lr"]
        for batch in torch.randperm(len(a), generator=generator).split(config["batch"]):
            opt.zero_grad()
            err = loss(model(a[batch]).flatten(), ay[batch])
            err.backward()
            opt.step()
            total += err.item() * len(batch)
        model.eval()
        with torch.no_grad():
            validation = loss(model(b).flatten(), by).item()
        if not np.isfinite(validation) or not np.isfinite(total):
            raise ArithmeticError(f"Non-finite loss in {run} at epoch {epoch}")
        history.append(dict(epoch=epoch, train_loss=total/len(a), validation_loss=validation, lr=lr))
        if config["scheduler"]:
            scheduler.step(validation)
        if validation < best - 1e-6:
            best, best_epoch = validation, epoch
            weights = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 15:
            break
    model.load_state_dict(weights)
    with torch.no_grad():
        score = model(transform(assess)).flatten().sigmoid().numpy()
    elapsed = time.perf_counter() - start
    metadata = dict(best_epoch=best_epoch, stopped_epoch=epoch, seconds=elapsed,
                    parameters=sum(p.numel() for p in model.parameters()), input_width=widths[0],
                    split_seed=split_seed, weight_seed=weight_seed,
                    status="epoch_cap" if epoch == 150 else "early_stop")
    save(f"results/training/{run}.csv", pd.DataFrame(history).assign(**metadata))
    if run.startswith("locked_"):
        (ROOT/"models").mkdir(exist_ok=True)
        torch.save(weights, ROOT/"models"/f"{run}.pt")
        joblib.dump(dict(preprocessor=pre, low=low, high=high, config=config, **metadata),
                    ROOT/"models"/f"{run}.joblib")
    return score, metadata


def metrics(y, score):
    pred = np.asarray(score) >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(AP=average_precision_score(y, score), ROC_AUC=roc_auc_score(y, score),
                Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Sensitivity=recall_score(y, pred), Specificity=tn/(tn+fp),
                F1=f1_score(y, pred), Balanced_Accuracy=balanced_accuracy_score(y, pred))


def folds(rows, y, seed):
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(rows, y.loc[rows]):
        yield rows[a], rows[b]


def search(x, y, rows, stage, seed):
    records = []
    for inner, (fit, assess) in enumerate(folds(rows, y, seed), 1):
        for group, configs in zip(("Manual", "Random"), candidates()):
            for c in configs:
                result, meta = train(x, y, fit, assess, c, seed + inner, seed + 10*inner,
                                     f"{stage}_inner{inner}_{c['id']}")
                records.append(dict(stage=stage, group=group, candidate=c["id"], inner=inner,
                                    AP=average_precision_score(y.loc[assess], result), **meta))
    frame = pd.DataFrame(records)
    summary = frame.groupby(["group", "candidate"]).agg(
        mean_AP=("AP", "mean"), sd_AP=("AP", "std"), seconds=("seconds", "sum"),
        mean_best_epoch=("best_epoch", "mean")).reset_index()
    return frame, summary.sort_values(["mean_AP", "candidate"], ascending=[False, True])


def config_by_id(candidate):
    return next(c for group in candidates() for c in group if c["id"] == candidate)


def reference(x, y, fit, assess, name, save_model=False):
    start = time.perf_counter()
    x, pre, _, _ = preprocess(x, fit, {"scaler": "Standard", "clip": False})
    model = (DummyClassifier(strategy="prior") if name == "Dummy"
             else LogisticRegression(max_iter=1000, random_state=BASE_SEED))
    model.fit(pre.transform(x.loc[fit]), y.loc[fit])
    score = model.predict_proba(pre.transform(x.loc[assess]))[:, 1]
    if save_model:
        joblib.dump(dict(preprocessor=pre, estimator=model), ROOT/"models"/f"{name}.joblib")
    return score, time.perf_counter()-start


def summarize(frame, key):
    numbers = frame[list(metrics([0, 1], [0, 1])) + ["seconds"]]
    return numbers.groupby(frame[key]).agg(["mean", "std"]).pipe(
        lambda v: v.set_axis([f"{m}_{s}" for m, s in v.columns], axis=1)).reset_index()


def outer():
    x, y, dev, _ = split()
    searches, scores, summaries = [], [], []
    for fold, (fit, assess) in enumerate(folds(dev, y, BASE_SEED+3), 1):
        frame, summary = search(x, y, fit, f"outer{fold}", BASE_SEED+20*fold)
        searches.append(frame)
        summaries.append(summary.assign(outer_fold=fold))
        for group in ("Manual", "Random"):
            candidate = summary.loc[summary.group.eq(group), "candidate"].iloc[0]
            score, meta = train(x, y, fit, assess, config_by_id(candidate),
                                BASE_SEED+200+fold, BASE_SEED+300+fold, f"outer{fold}_{group}")
            scores.append(dict(fold=fold, method=group, candidate=candidate,
                               seconds=meta["seconds"], **metrics(y.loc[assess], score)))
        for name in ("Dummy", "LogisticRegression"):
            score, duration = reference(x, y, fit, assess, name)
            scores.append(dict(fold=fold, method=name, candidate=name, seconds=duration,
                               **metrics(y.loc[assess], score)))
    frame = pd.concat(searches)
    summary = pd.concat(summaries)
    for group in ("Manual", "Random"):
        save(f"results/{group.lower()}_search_results.csv", frame.loc[frame.group.eq(group)])
    save("results/outer_inner_summary.csv", summary)
    scores = pd.DataFrame(scores)
    save("results/outer_cv_results.csv", scores)
    save("results/outer_cv_summary.csv", summarize(scores, "method"))
    scores.pivot(index="fold", columns="method", values="AP").plot.bar(ylabel="Average precision")
    plot("outer_ap_comparison.png")


def lock():
    x, y, dev, test = split()
    frame, summary = search(x, y, dev, "final", BASE_SEED+400)
    save("results/final_inner_folds.csv", frame)
    save("results/final_inner_search.csv", summary)
    c = config_by_id(summary.iloc[0].candidate)
    predictions = pd.DataFrame({"row_index": test})
    model_metadata = []
    for seed in SEEDS:
        score, meta = train(x, y, dev, test, c, BASE_SEED+500, seed, f"locked_{seed}")
        predictions[f"prob_{seed}"] = score
        predictions[f"pred_{seed}"] = (score >= .5).astype(int)
        model_metadata.append(meta)
    reference_seconds = {}
    for name in ("Dummy", "LogisticRegression"):
        score, seconds = reference(x, y, dev, test, name, save_model=True)
        predictions[f"prob_{name}"] = score
        predictions[f"pred_{name}"] = (score >= .5).astype(int)
        reference_seconds[name] = seconds
    save("results/test_predictions_locked.csv", predictions)
    save("results/final_hyperparameters.json", dict(config=c, primary_seed=SEEDS[0],
         threshold=.5, selection="highest mean inner AP; ties by candidate identifier",
         models=model_metadata, reference_seconds=reference_seconds))
    summary.sort_values("candidate").plot.bar(x="candidate", y="mean_AP", yerr="sd_AP",
                                             legend=False, ylabel="Development inner AP")
    plot("final_candidate_comparison.png")


def check_lock():
    def git(*args, text=True):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=text).strip()
    remote = git("remote", "get-url", "origin")
    allowed = ("https://github.com/JustLotus/YapayS-Vize", "git@github.com:JustLotus/YapayS-Vize")
    if remote.removesuffix(".git") not in allowed:
        raise RuntimeError("origin must be the assignment GitHub repository")
    git("fetch", "--quiet", "origin", "main")
    commit = git("log", "-1", "--format=%H", "FETCH_HEAD", "--",
                 "results/test_predictions_locked.csv")
    if not commit:
        raise RuntimeError("Push the prediction lock before opening test labels")
    for name in ("results/test_predictions_locked.csv", "results/final_hyperparameters.json",
                 "src/experiment.py"):
        committed = git("show", f"{commit}:{name}", text=False).replace(b"\r\n", b"\n").strip()
        current = (ROOT/name).read_bytes().replace(b"\r\n", b"\n").strip()
        if current != committed:
            raise RuntimeError(f"Local {name} differs from the published lock")
    return commit


def evaluate():
    commit = check_lock()
    x, y, dev, test = split()
    locked = csv("test_predictions_locked.csv")
    if not np.array_equal(locked.row_index.to_numpy(), test):
        raise ValueError("Locked rows differ from the independent split")
    target = y.loc[test].to_numpy()
    save("results/dataset_split.csv", split_table(y, dev, test, reveal=True))
    config = json.loads((ROOT/"results/final_hyperparameters.json").read_text(encoding="utf-8"))
    methods = {"MLP": locked[f"prob_{SEEDS[0]}"].to_numpy(),
               **{name: locked[f"prob_{name}"].to_numpy() for name in ("LogisticRegression", "Dummy")}}
    seconds = dict(MLP=config["models"][0]["seconds"], **config["reference_seconds"])
    save("results/test_metrics.csv", pd.DataFrame([
        dict(method=name, seconds=seconds[name], **metrics(target, score)) for name, score in methods.items()]))
    seed_rows = pd.DataFrame([dict(seed=seed, group="MLP", seconds=meta["seconds"],
                                  **metrics(target, locked[f"prob_{seed}"]))
                              for seed, meta in zip(SEEDS, config["models"])])
    save("results/seed_stability_results.csv", seed_rows.drop(columns="group"))
    save("results/seed_stability_summary.csv", summarize(seed_rows, "group"))
    rng = np.random.default_rng(BASE_SEED+600)
    mlp = methods["MLP"]
    samples = []
    while len(samples) < 1000:
        idx = rng.integers(0, len(test), len(test))
        yy, ss = target[idx], mlp[idx]
        if len(np.unique(yy)) == 2:
            samples.append([roc_auc_score(yy, ss), average_precision_score(yy, ss), f1_score(yy, ss>=.5)])
    intervals = np.percentile(samples, [2.5, 97.5], axis=0)
    save("results/bootstrap_intervals.json", {metric: list(bounds) for metric, bounds
         in zip(("ROC_AUC", "AP", "F1"), intervals.T)})
    locked_bytes = (ROOT/"results/test_predictions_locked.csv").read_bytes().replace(b"\r\n", b"\n")
    save("results/evaluation_provenance.json", dict(prediction_lock_commit=commit, threshold=.5,
         prediction_sha256=hashlib.sha256(locked_bytes).hexdigest(),
         bootstrap_replicates=1000, bootstrap_seed=BASE_SEED+600))
    pred = mlp >= .5
    error = np.where(pred == target, "correct", np.where(pred, "FP", "FN"))
    save("results/error_type_counts.csv", pd.Series(error).value_counts().rename_axis("type").reset_index(name="count"))
    mask = error != "correct"
    save("results/test_errors.csv", x.loc[test[mask]].rename_axis("row_index").reset_index().assign(
        error=error[mask], y=target[mask], probability=mlp[mask]))
    for name, score in methods.items():
        if name == "Dummy":
            continue
        matrix = confusion_matrix(target, score >= .5)
        plt.imshow(matrix, cmap="Blues")
        for i, j in np.ndindex(matrix.shape):
            plt.text(j, i, str(matrix[i, j]), ha="center", color="white" if matrix[i,j]>matrix.max()/2 else "black")
        plt.xticks([0,1]); plt.yticks([0,1])
        plt.xlabel("Prediction"); plt.ylabel("Observed"); plt.title(name)
        plot(f"test_confusion_{name}.png")
    for chart in ("roc", "pr"):
        for name, score in methods.items():
            if chart == "roc":
                fpr, tpr, _ = roc_curve(target, score)
                plt.plot(fpr, tpr, label=name)
            else:
                precision, recall, _ = precision_recall_curve(target, score)
                plt.plot(recall, precision, label=name)
        plt.xlabel("False positive rate" if chart == "roc" else "Recall")
        plt.ylabel("True positive rate" if chart == "roc" else "Precision")
        plt.legend()
        plot(f"test_{chart}.png")
    history = csv(f"training/locked_{SEEDS[0]}.csv")
    history.plot(x="epoch", y=["train_loss", "validation_loss"], ylabel="BCE")
    plot("final_learning_curves.png")
    history.plot(x="epoch", y="lr", ylabel="Learning rate", legend=False, logy=True)
    plot("final_learning_rate.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "outer", "lock", "evaluate"))
    globals()[parser.parse_args().stage]()
