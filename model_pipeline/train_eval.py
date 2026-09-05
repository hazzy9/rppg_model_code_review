# train_eval.py

"""
CV training/evaluation runner. Matches manuscript: 5-fold Stratified Group
K-Fold, repeated 4x with prediction-averaging across repeats; Adam + L1
loss + gradient norm clipping.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import TinyCNN
from features import make_tfm_stack_from_list, build_meta_planes_for_rows
from fpg_utils import ImgOnlyDataset, SimpleScaler, set_seeds_all

import os
import time
from sklearn.model_selection import StratifiedGroupKFold

from tuning import run_hpo


def _grouped_inner_split(tr_idx, licenses_all, val_frac, rng_local):
    """Splits tr_idx into (inner_train, inner_val), keeping every license
    entirely on one side -- a subject's rows can never be split across
    the two. To prevent leakage of data of the same subject into both train and validation groups"""
    
    inner_licenses = licenses_all[tr_idx]
    unique_lic = np.unique(inner_licenses)
    rng_local.shuffle(unique_lic)
    n_val_lic = max(1, int(round(val_frac * len(unique_lic))))
    val_lic_set = set(unique_lic[:n_val_lic])
    is_val = np.isin(inner_licenses, list(val_lic_set))
    return tr_idx[~is_val], tr_idx[is_val]



def _train_one_fold(
    df_outer_train: pd.DataFrame,
    df_outer_test: pd.DataFrame,
    channels_img: list,
    hp: dict,
    seed: int,
    inner_val_frac: float = 0.15,
    device=None,
):
    """
    Train and evaluate the model for one outer cross-validation fold, 
    using a subject-disjoint inner validation split for early stopping.

    """
    set_seeds_all(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
 
    groups_all = df_outer_train["license"].astype(str).to_numpy()
    all_idx = np.arange(len(df_outer_train))
    inner_tr_idx, inner_va_idx = _grouped_inner_split(all_idx, groups_all, inner_val_frac, rng)
 
    assert not set(groups_all[inner_tr_idx]) & set(groups_all[inner_va_idx]), \
        "license leaked: inner-train/inner-val"
 
    df_tr = df_outer_train.iloc[inner_tr_idx].reset_index(drop=True)
    df_va = df_outer_train.iloc[inner_va_idx].reset_index(drop=True)
    df_te = df_outer_test.reset_index(drop=True)
 
    H = W = 64
    X_img_tr = np.stack([make_tfm_stack_from_list(r, channels_img, (H, W)) for _, r in df_tr.iterrows()]).astype(np.float32)
    X_img_va = np.stack([make_tfm_stack_from_list(r, channels_img, (H, W)) for _, r in df_va.iterrows()]).astype(np.float32)
    X_img_te = np.stack([make_tfm_stack_from_list(r, channels_img, (H, W)) for _, r in df_te.iterrows()]).astype(np.float32)
 
    age_scaler = SimpleScaler().fit(df_tr["age"].to_numpy())  # inner-train only
    X_meta_tr = build_meta_planes_for_rows(df_tr, age_scaler, H=H, W=W)
    X_meta_va = build_meta_planes_for_rows(df_va, age_scaler, H=H, W=W)
    X_meta_te = build_meta_planes_for_rows(df_te, age_scaler, H=H, W=W)
 
    Xtr = np.concatenate([X_img_tr, X_meta_tr], axis=-1)
    Xva = np.concatenate([X_img_va, X_meta_va], axis=-1)
    Xte = np.concatenate([X_img_te, X_meta_te], axis=-1)
 
    ytr = df_tr["fpg"].astype(float).to_numpy()
    yva = df_va["fpg"].astype(float).to_numpy()
    yte = df_te["fpg"].astype(float).to_numpy()
 
    BATCH_TR = int(hp["batch_size"])
    BATCH_VA = int(max(BATCH_TR, 64))
    LR = float(hp["lr"])
    WD = float(hp["weight_decay"])
    P_DROP = float(hp["dropout_p"])
    GRAD_CLIP = float(hp["grad_clip_max_norm"])
    PATIENCE = int(hp["patience"])
    MAX_EPOCHS = int(hp["max_epochs"])
 
    g = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
    train_loader = DataLoader(ImgOnlyDataset(Xtr, ytr), batch_size=BATCH_TR, shuffle=True, generator=g)
    val_loader = DataLoader(ImgOnlyDataset(Xva, yva), batch_size=BATCH_VA, shuffle=False)
    test_loader = DataLoader(ImgOnlyDataset(Xte, yte), batch_size=BATCH_VA, shuffle=False)
 
    in_ch = Xtr.shape[-1]
    model = TinyCNN(in_ch=in_ch, p_drop=P_DROP).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.L1Loss()
 
    best, best_state, waited, best_epoch = float("inf"), None, 0, 0
 
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optim.step()
 
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += lossf(model(xb), yb).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)
 
        if va_loss < best - 1e-6:
            best, waited, best_epoch = va_loss, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
 
    if best_state is not None:
        model.load_state_dict(best_state)
 
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred = np.vstack(preds).ravel()
 
    return {
        "model_state": best_state,
        "in_ch": in_ch,
        "p_drop": P_DROP,
        "y_true": yte,
        "y_pred": y_pred,
        "license": df_te["license"].astype(str).to_numpy(),
        "fpg_category": (df_te["fpg_category"].astype(str).to_numpy()
                          if "fpg_category" in df_te.columns else None),
        "best_inner_val_mae": best,
        "best_epoch": best_epoch,
    }

# run nested cv


def run_nested_cv(
    df_proc: pd.DataFrame,
    channels_img: list,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    n_hpo_trials: int = 80,
    base_hp: dict = None,
    inner_val_frac: float = 0.15,
    hpo_seed_base: int = 123,
    outer_seed: int = 1354,
    device=None,
    save_model_dir: str = None,
    
):
    """
    Full nested cross-validation. For each of n_outer_folds folds:
        1. split off outer-test 
        2. run_hpo on outer-train only
        3. _train_one_fold with best_hp -> trained model + outer-test predictions
        4. predictions recorded -- first and only use of outer-test
    
    No repeat dimension (n_repeats=1) -- if repeated nested CV is ever
    wanted, call this function multiple times with different outer_seed values.
    """
    y_all = df_proc["fpg"].astype(float).to_numpy()
    y_strat = (
        df_proc["fpg_category"].astype(str).to_numpy()
        if "fpg_category" in df_proc.columns
        else pd.cut(
            df_proc["fpg"].astype(float),
            bins=[-np.inf, 100, 126, np.inf],
            right=False,
            labels=["<100", "100-125", ">=126"],
        ).astype(str).to_numpy()
    )
    groups = df_proc["license"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=outer_seed)
    X_dummy = np.zeros(len(y_all))

    fold_results = []
    hp_per_fold = []
    studies = []

    for fold, (outer_tr_idx, outer_te_idx) in enumerate(cv.split(X_dummy, y=y_strat, groups=groups), 1):
        assert not set(groups[outer_tr_idx]) & set(groups[outer_te_idx]), \
            f"license leaked: outer-train/outer-test, fold {fold}"

        df_outer_train = df_proc.iloc[outer_tr_idx].reset_index(drop=True)
        df_outer_test = df_proc.iloc[outer_te_idx].reset_index(drop=True)

        print(f"\n=== Outer fold {fold}/{n_outer_folds} ===")
        print(f"outer-train: {len(df_outer_train)} rows, {df_outer_train['license'].nunique()} subjects")
        print(f"outer-test:  {len(df_outer_test)} rows, {df_outer_test['license'].nunique()} subjects")

        t0 = time.time()
        study, best_hp = run_hpo(
            df_train=df_outer_train,
            images_train=None,
            channels_img=channels_img,
            base_hp=base_hp,
            n_trials=n_hpo_trials,
            n_inner_folds=n_inner_folds,
            seed=hpo_seed_base + fold,
        )
        t_hpo = time.time() - t0
        print(f"[fold {fold}] HPO done in {t_hpo/60:.1f} min | best val_mae={study.best_value:.3f}")

        hp_record = dict(best_hp)
        hp_record["outer_fold"] = fold
        hp_per_fold.append(hp_record)
        studies.append(study)

        t0 = time.time()
        result = _train_one_fold(
            df_outer_train=df_outer_train,
            df_outer_test=df_outer_test,
            channels_img=channels_img,
            hp=best_hp,
            seed=outer_seed + fold,
            inner_val_frac=inner_val_frac,
            device=device,
        )
        t_train = time.time() - t0
        print(f"[fold {fold}] Final training done in {t_train/60:.1f} min | "
              f"best inner-val MAE={result['best_inner_val_mae']:.3f} (stopped at epoch {result['best_epoch']})")

        result["outer_fold"] = fold
        fold_results.append(result)

        if save_model_dir is not None:
            os.makedirs(save_model_dir, exist_ok=True)
            torch.save(
                {
                    "outer_fold": fold,
                    "state_dict": result["model_state"],
                    "in_ch": result["in_ch"],
                    "p_drop": result["p_drop"],
                    "hp": best_hp,
                },
                os.path.join(save_model_dir, f"outer_fold{fold}.pth"),
            )

    details_rows = []
    for r in fold_results:
        n = len(r["y_true"])
        details_rows.append(pd.DataFrame({
            "outer_fold": r["outer_fold"],
            "license": r["license"],
            "fpg_category": r["fpg_category"] if r["fpg_category"] is not None else [None] * n,
            "y_true": r["y_true"],
            "y_pred": r["y_pred"],
        }))
    details_df = pd.concat(details_rows, ignore_index=True)

    assert len(details_df) == len(df_proc), (
        f"pooled predictions cover {len(details_df)} rows, expected {len(df_proc)} -- "
        "some rows were never in any outer-test fold, or appeared more than once"
    )
    assert details_df["license"].nunique() == df_proc["license"].nunique(), (
        "pooled predictions don't cover every subject in df_proc"
    )

    hp_per_fold_df = pd.DataFrame(hp_per_fold)

    return {
        "details": details_df,
        "hp_per_fold": hp_per_fold_df,
        "studies": studies,
    }


# runs full nested CV + evaluation
def run_final_baseline_3tfm(
        df_proc : pd.DataFrame, 
        channels_img : list = None,
        n_outer_folds: int = 5,
        n_inner_folds: int = 3,
        n_hpo_trials: int = 80,
        base_hp: dict = None,
        hpo_seed_base: int = 123,
        outer_seed: int = 1354,
        device = None,
        save_csv: bool = False,
        save_dir : str = "final_baseline_outputs",
        save_model_dir : str = None,
):
    from eval import compute_icgm_style_metrics_v2, compute_iso_style_metrics, clarke_error_grid
    from fpg_utils import compute_paper_metrics

    if channels_img is None:
        channels_img = ["crcb", "pos", "pbv"]
    
    model_tag = f"nested_cv_{'+'.join(channels_img)}"

    #nested CV run
    nested_out = run_nested_cv(
        df_proc=df_proc,
        channels_img=channels_img,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        n_hpo_trials=n_hpo_trials,
        base_hp = base_hp,
        hpo_seed_base=hpo_seed_base,
        outer_seed=outer_seed,
        device=device,
        save_model_dir=save_model_dir
    )

    output_df = nested_out["details"].copy()
    output_df["model"] = model_tag

    y_true = output_df["y_true"].to_numpy()
    y_pred = output_df["y_pred"].to_numpy()

    #overall pooled metrics
    output_metrics = compute_paper_metrics(y_true, y_pred)

    # per outer fold breakdown
    per_fold_rows = []
    for fold, sub in output_df.groupby("outer_fold"):
        row = {"outer_fold" : fold, "n": len(sub)}
        row.update(compute_paper_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()))
        per_fold_rows.append(row)
    per_fold_metrics = pd.DataFrame(per_fold_rows).sort_values("outer_fold").reset_index(drop=True)

    # ---- clinical accuracy tables, on the pooled predictions ----
    icgm_df = compute_icgm_style_metrics_v2(output_df, model_tag, pred_col="y_pred")
    iso_df = compute_iso_style_metrics(output_df, model_tag, pred_col="y_pred", ref_col="y_true", twenty_break=100)
    clarke = clarke_error_grid(y_true, y_pred)

    if save_csv:
        os.makedirs(save_dir, exist_ok=True)
        output_df.to_csv(f"{save_dir}/details_pooled.csv", index=False)
        nested_out["hp_per_fold"].to_csv(f"{save_dir}/hp_per_fold.csv", index=False)
        per_fold_metrics.to_csv(f"{save_dir}/per_fold_metrics.csv", index=False)
        icgm_df.to_csv(f"{save_dir}/icgm_metrics.csv", index=False)
        iso_df.to_csv(f"{save_dir}/iso_metrics.csv", index=False)
        pd.DataFrame([{"metric": k, "value": v} for k, v in output_metrics.items()]).to_csv(
            f"{save_dir}/paper_metrics.csv", index=False)
        pd.DataFrame([{"zone": k, "value": v} for k, v in clarke.items()]).to_csv(
            f"{save_dir}/clarke_grid.csv", index=False)
        print(f"[Final] Saved CSVs -> {save_dir}/")

    return {
        "details": output_df,
        "hp_per_fold": nested_out["hp_per_fold"],
        "per_fold_metrics": per_fold_metrics,
        "paper_metrics": output_metrics,
        "icgm": icgm_df,
        "iso": iso_df,
        "clarke": clarke,
        "studies": nested_out["studies"],
    }


