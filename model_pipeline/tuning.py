#tuning

import json 
import numpy as np 
import pandas as pd 
import optuna
import torch 
from torch import nn 
from torch.utils.data import DataLoader 
from sklearn.model_selection import StratifiedGroupKFold

#modules import
from model import TinyCNN 
from features import make_tfm_stack_from_list, build_meta_planes_for_rows
from fpg_utils import ImgOnlyDataset, SimpleScaler, set_seeds_all



def _maybe_slice_precomputed(images, needed_channels, tr_idx, va_idx):

    if images is None:
        return None, None

    if hasattr(images, "channels"):         
        if tuple(images.channels) == tuple(needed_channels):
            data = images.data if hasattr(images, "data") else images
            return data[tr_idx], data[va_idx]

    if isinstance(images, dict) and "channels" in images and "data" in images:
        if tuple(images["channels"]) == tuple(needed_channels):
            return images["data"][tr_idx], images["data"][va_idx]

    return None, None


def _compute_hpo_objective(y_true, y_pred):

    # validation MAE
    val_mae = float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))
    return val_mae, {"val_mae": val_mae}


def _cv_once_with_cfg(df_use: pd.DataFrame, images, cfg: dict, n_folds=3, seed=1354, model_seed = None):
    if model_seed is None:
        model_seed = seed
    
    y_all = df_use["fpg"].astype(float).to_numpy()

    y_strat = (
        df_use["fpg_category"].astype(str).to_numpy()
        if "fpg_category" in df_use.columns
        else pd.cut(
            df_use["fpg"].astype(float),
            bins=[-np.inf, 100, 126, np.inf],
            right=False,
            labels=["<100", "100-125", ">=126"],
        ).astype(str).to_numpy()
    )
    groups = (
        df_use["license"].astype(str).to_numpy()
        if "license" in df_use.columns
        else np.arange(len(df_use))
    )

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_dummy = np.zeros(len(y_all))

    BATCH_TR = int(cfg.get("batch_size", 32))
    BATCH_VA = int(max(BATCH_TR, 64))
    LR = float(cfg.get("lr", 1e-3))
    WD = float(cfg.get("weight_decay", 1e-4))
    MAX_EPOCHS = int(cfg.get("max_epochs", 60))
    PATIENCE = int(cfg.get("patience", 10))
    P_DROP = float(cfg.get("dropout_p", 0.15))
    GRAD_CLIP = float(cfg.get("grad_clip_max_norm", 100.0)) # based on prior test that checks rough range of possible values 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels_trial = cfg["channels_img"]
    H = W = 64

    fold_scores, fold_parts = [], []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_dummy, y=y_strat, groups=groups), 1):
        set_seeds_all(model_seed + fold)
        df_tr = df_use.iloc[tr_idx].reset_index(drop=True)
        df_va = df_use.iloc[va_idx].reset_index(drop=True)

        X_img_tr, X_img_va = _maybe_slice_precomputed(images, channels_trial, tr_idx, va_idx)
        if X_img_tr is None:
            X_img_tr = np.stack([make_tfm_stack_from_list(r, channels_trial, (H, W)) for _, r in df_tr.iterrows()])
            X_img_va = np.stack([make_tfm_stack_from_list(r, channels_trial, (H, W)) for _, r in df_va.iterrows()])

        age_scaler = SimpleScaler().fit(df_tr["age"].to_numpy())  # this fold's training partition only
        X_meta_tr = build_meta_planes_for_rows(df_tr, age_scaler, H=H, W=W)
        X_meta_va = build_meta_planes_for_rows(df_va, age_scaler, H=H, W=W)

        Xtr = np.concatenate([X_img_tr, X_meta_tr], axis=-1)
        Xva = np.concatenate([X_img_va, X_meta_va], axis=-1)
        ytr, yva = y_all[tr_idx], y_all[va_idx]

        train_loader = DataLoader(ImgOnlyDataset(Xtr, ytr), batch_size=BATCH_TR, shuffle=True, num_workers=0)
        val_loader = DataLoader(ImgOnlyDataset(Xva, yva), batch_size=BATCH_VA, shuffle=False, num_workers=0)

        lossf = nn.L1Loss()  # fixed -- not searched
        model = TinyCNN(in_ch=Xtr.shape[-1], p_drop=P_DROP).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)  # matches final training loop

        best, waited, best_state, best_epoch = float("inf"), 0, None, 0

        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                opt.step()

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    va_loss += lossf(model(xb), yb).item() * xb.size(0)
            va_loss /= len(val_loader.dataset)

            if va_loss < best - 1e-6:
                best, waited, best_epoch = va_loss, 0, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    break

        model.load_state_dict(best_state)

        preds = []
        model.eval()
        with torch.no_grad():
            for xb, _ in val_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
        y_pred = np.vstack(preds).ravel()

        s, p = _compute_hpo_objective(yva, y_pred)
        fold_scores.append(s)
        fold_parts.append(p)
        p["best_epoch"] = best_epoch

    final_score = float(np.nanmean(fold_scores))
    if not np.isfinite(final_score):
        final_score = 1e6

    summary = {k: float(np.nanmean([fp[k] for fp in fold_parts])) for k in fold_parts[0].keys()}
    summary["best_epoch"] = float(np.nanmedian([fp["best_epoch"] for fp in fold_parts]))
    return final_score, summary


def run_hpo(
    df_train: pd.DataFrame,
    images_train,
    channels_img: list,
    base_hp: dict = None,
    n_trials: int = 80,
    n_inner_folds: int = 3,
    timeout_hours: float = None,
    seed: int = 123,
):
    """
    Runs one independent Optuna search 
    (default: 80 trials, TPE) on df_train/images_train only 
    -- the caller is responsible for ensuring these contain no outer-test data.

    Returns (study, best_hp) -- best_hp is a plain dict, ready to pass directly to train_eval._train_one_fold(). 
    File I/O is a separate, optional step (see save_hp below); this function never writes to disk.
    """
    base_hp = dict(base_hp) if base_hp else {}

    if images_train is not None:
        setattr(images_train, "channels", tuple(channels_img))

    def objective(trial: optuna.Trial):
        trial_seed = seed + (trial.number * 50) 
        set_seeds_all(trial_seed)
        cfg = dict(base_hp)

        cfg["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
        cfg["dropout_p"] = trial.suggest_float("dropout_p", 0.05, 0.20)
        cfg["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
        cfg["patience"] = trial.suggest_categorical("patience", [8, 10, 12])
        cfg["max_epochs"] = trial.suggest_categorical("max_epochs", [60, 80])
        cfg["grad_clip_max_norm"] = trial.suggest_float("grad_clip_max_norm", 10.0, 750.0, log=True)
        cfg["channels_img"] = list(channels_img)

        score, parts = _cv_once_with_cfg(df_train, images_train, cfg, n_folds=n_inner_folds, seed=seed, model_seed= trial_seed)
        trial.set_user_attr("val_mae", parts["val_mae"])
        trial.set_user_attr("median_best_epoch", parts["best_epoch"])
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None if timeout_hours is None else int(timeout_hours * 3600),
        gc_after_trial=True,
    )

    best = study.best_params
    median_best_epoch = study.best_trial.user_attrs["median_best_epoch"]

    print(f"Best params: {best} | best val_mae: {study.best_value:.4f} | "
          f"median_best_epoch: {median_best_epoch:.1f}")

    best_hp = dict(base_hp)
    best_hp.update({
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "dropout_p": best["dropout_p"],
        "batch_size": best["batch_size"],
        "patience": best["patience"],  # kept for record; not used by the final refit itself
        "grad_clip_max_norm": best["grad_clip_max_norm"],
        "final_refit_epochs": int(round(median_best_epoch)),  # used by _train_one_fold
        "max_epochs": base_hp.get("max_epochs", 120),  # record of the search's per-trial ceiling
        "loss_name": "mae",
        "seed": base_hp.get("seed", 1354),
        "channels_img": list(channels_img),
    })

    return study, best_hp


def load_hp(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_hp(hp: dict, path: str):
    with open(path, "w") as f:
        json.dump(hp, f, indent=2)