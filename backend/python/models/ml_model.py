import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, ndcg_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn import set_config
from typing import Union

from lightgbm import LGBMRegressor, LGBMRanker, early_stopping, log_evaluation
import shap
import optuna

# Import feature engineering utilities
from data.create_features import load_feature_dataframe

# ------------------
# Configuration
# ------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
set_config(enable_metadata_routing=True)
HOLD_DAYS_DEFAULT = 2
TOP_K_DEFAULT = 5
VAL_FRAC_DEFAULT = 0.2
TEST_FRAC_DEFAULT = 0.2

RAW_FEATURE_COLS = [
    'delta',
    'moneyness',
    'tlt_return_5d',
    'delta_iv',
    'stoch_d',
    'stk_range_pct_zscore',
    'implied_volatility',
]

LGB_EARLY_STOPPING = [
    early_stopping(stopping_rounds=50),
    log_evaluation(period=0)    # silence per-iteration logs
]

# ------------------
# Label Generation
# ------------------

def generate_return_label(df: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """
    Generate forward percentage returns over `hold_days` for each option.
    Adds column 'ret_forward'.
    """
    pivot = df.pivot(
        index='trade_date',
        columns='option_symbol',
        values='mid_price'
    )
    forward = pivot.pct_change(periods=hold_days, fill_method=None).shift(-hold_days)
    ret_df = (
        forward.reset_index()
               .melt(id_vars='trade_date', var_name='option_symbol', value_name='ret_forward')
    )
    df_labeled = df.merge(ret_df, on=['trade_date', 'option_symbol'], how='inner')
    return df_labeled.dropna(subset=['ret_forward'])

# ------------------
# Train/Test Split
# ------------------

def split_data(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by trade_date into train/val/test sets without leakage.

    Returns:
        train_df, val_df, test_df
    """
    dates = df['trade_date'].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * (1 - val_frac - test_frac))]
    val_end = dates[int(n * (1 - test_frac))]

    train_df = df[df['trade_date'] <= train_end]
    val_df = df[(df['trade_date'] > train_end) & (df['trade_date'] <= val_end)]
    test_df = df[df['trade_date'] > val_end]

    logging.info(
        f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df

# ------------------
# Model Training
# ------------------

def tune_best_base_model_params(
    X_train, y_train,
    X_val,   y_val,
    force_retrain
):
    """
    Uses Optuna to minimize val MSE for the base regressor.
    """
    def objective(trial):
        # 1) sample hyperparameters
        params = {
            'num_leaves':       trial.suggest_int('num_leaves', 31, 150),
            'max_depth':        trial.suggest_int('max_depth', 5, 12),
            'learning_rate':    trial.suggest_loguniform('learning_rate', 0.005, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'lambda_l1':        trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
            'lambda_l2':        trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
        }
        model = LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            callbacks=LGB_EARLY_STOPPING
        )
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    def search_fn():
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=40)  # bump trials as you like
        return study.best_params

    return get_or_tune_params(
        "base_mse",
        search_fn,
        "base_model_params.json",
        force_retrain
    )

def get_or_tune_params(name: str, param_search_fn, params_path: str, force_retrain: bool):
    """
    Load best‐params from JSON unless force_retrain=True, in which case we always re-run search.
    """
    if not force_retrain and os.path.exists(params_path):
        with open(params_path, 'r') as f:
            best = json.load(f)
        logging.info(f"Loaded {name} params from {params_path}")
    else:
        best = param_search_fn()
        with open(params_path, 'w') as f:
            json.dump(best, f, indent=2)
        logging.info(f"Saved {name} params to {params_path}")
    return best

def tune_best_quantile_params(
    X_train, y_train,
    X_val,   y_val,
    α,
    force_retrain
):
    """
    Uses Optuna to minimize validation MSE for a single quantile α.
    """
    def objective(trial):
        # sample hyperparameters
        params = {
            'objective': 'quantile',
            'alpha': α,
            'num_leaves':       trial.suggest_int('num_leaves', 31, 150),
            'max_depth':        trial.suggest_int('max_depth', 5, 12),
            'learning_rate':    trial.suggest_loguniform('learning_rate', 0.005, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'lambda_l1':        trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
            'lambda_l2':        trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
            'min_gain_to_split':trial.suggest_uniform('min_gain_to_split', 0.0, 0.5),
        }
        model = LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            callbacks=LGB_EARLY_STOPPING
        )
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    def search_fn():
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        return study.best_params

    return get_or_tune_params(
        f"quantile_{α}",
        search_fn,
        f"quantile_{int(α*100)}_params.json",
        force_retrain
    )

def train_quantile_models_persisted(X_train, y_train,
                                   X_val,   y_val,
                                   quantiles,
                                   force_retrain):
    models = {}
    for α in quantiles:
        best = tune_best_quantile_params(
            X_train, y_train,
            X_val,   y_val,
            α, force_retrain
        )
        m = LGBMRegressor(objective='quantile', alpha=α,
                          random_state=RANDOM_SEED, n_jobs=-1, **best)
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            callbacks=LGB_EARLY_STOPPING
        )
        models[α] = m
    return models

def tune_best_ranker_params(
    X_train, y_train_rel, group_train,
    X_val,   y_val_rel,   group_val,
    force_retrain
):
    """
    Uses Optuna to maximize validation NDCG@5 for LambdaRank.
    """
    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric':    'ndcg',
            # smaller, shallower trees
            'num_leaves':       trial.suggest_int('num_leaves', 16, 64),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),

            # slower learning to avoid over-fit
            'learning_rate':    trial.suggest_loguniform('learning_rate', 0.005, 0.05),

            # stronger regularization
            'lambda_l1':        trial.suggest_loguniform('lambda_l1', 1e-2, 10.0),
            'lambda_l2':        trial.suggest_loguniform('lambda_l2', 1e-2, 10.0),

            # larger leaf minimum to smooth splits
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 500),

            # more aggressive feature/bagging fractions
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 0.9),
        }
        model = LGBMRanker(**params, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(
            X_train, y_train_rel,
            group=group_train,
            eval_set=[(X_val, y_val_rel)],
            eval_group=[group_val],
            eval_metric='ndcg',
            callbacks=LGB_EARLY_STOPPING
        )
        # compute mean NDCG@5 on validation
        preds = model.predict(X_val)
        # slice into lists by group_val
        start = 0
        scores = []
        for g in group_val:
            end = start + g
            true = y_val_rel[start:end]
            pred = preds[start:end]
            scores.append(ndcg_score([true], [pred], k=5))
            start = end
        return -np.mean(scores)

    def search_fn():
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        # Optuna returns the sign‐inverted NDCG, so we only need the params:
        return study.best_params

    return get_or_tune_params(
        "lambdarank",
        search_fn,
        "lambdarank_params.json",
        force_retrain
    )

def train_lambdarank_persisted(X_train, y_train, group_train,
                               X_val,   y_val,   group_val,
                               force_retrain):
    best = tune_best_ranker_params(
        X_train, y_train, group_train,
        X_val,   y_val,   group_val,
        force_retrain
    )
    ranker = LGBMRanker(objective='lambdarank', metric='ndcg',
                        random_state=RANDOM_SEED, n_jobs=-1, **best)
    ranker.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        eval_metric='ndcg',
        callbacks=LGB_EARLY_STOPPING
    )
    return ranker

class RankerAsRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps a fitted LGBMRanker so it looks like a regressor
    for use in StackingRegressor.
    """
    def __init__(self, ranker):
        self.ranker = ranker

    def fit(self, X, y=None):
        # no-op: we assume ranker is already trained
        return self

    def predict(self, X):
        return self.ranker.predict(X)

def tune_best_meta_params(
    preds_val: np.ndarray,
    y_val: np.ndarray,
    y_val_rel: np.ndarray,
    group_val: list[int],
    force_retrain: bool
) -> dict:
    """
    Uses Optuna to minimize a combined loss:
      L = λ * MSE + (1 - λ) * (1 - NDCG@5)
    Returns a dict with 'lambda_weight' plus the best LGBM hyperparams.
    """

    def objective(trial):
        # 1) sample the mix coefficient
        λ = trial.suggest_uniform('lambda_weight', 0.0, 1.0)

        # 2) sample tree hyperparameters
        params = {
            'num_leaves':       trial.suggest_int('num_leaves', 20, 100),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'learning_rate':    trial.suggest_loguniform('learning_rate', 0.005, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'lambda_l1':        trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
            'lambda_l2':        trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
        }

        # 3) train on preds_val → y_val (continuous)
        model = LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(
            preds_val, y_val,
            eval_set=[(preds_val, y_val)],
            eval_metric='l2',
            callbacks=LGB_EARLY_STOPPING
        )

        # 4) get meta‐predictions
        meta_pred = model.predict(preds_val)

        # 5) compute losses
        mse   = mean_squared_error(y_val, meta_pred)
        ndcg_ = evaluate_ndcg(meta_pred, None, y_val_rel, group_val, k=5)

        # 6) combined loss to minimize
        return λ * mse + (1 - λ) * (1 - ndcg_)

    def search_fn():
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=40)
        return study.best_trial.params  # includes 'lambda_weight'

    return get_or_tune_params(
        "meta_learner",
        search_fn,
        "meta_params.json",
        force_retrain
    )

def evaluate_ndcg(
    model_or_preds,            # either a fitted model with .predict, or an array of preds
    X: pd.DataFrame,           # if model, the features to predict on
    y: Union[pd.Series, np.ndarray],  # the true labels (flat)
    groups: list[int],         # list of group sizes in the same order as y
    k: int = 5
) -> float:
    """
    Compute mean NDCG@k across the given groups.
    If model_or_preds has a .predict method, calls it on X; otherwise treats it as preds array.
    """
    # 1) get predictions
    if hasattr(model_or_preds, "predict"):
        preds = model_or_preds.predict(X)
    else:
        preds = np.asarray(model_or_preds)

    # 2) slice into groups
    true_groups = []
    pred_groups = []
    start = 0
    for g in groups:
        end = start + g
        true_groups.append(y[start:end])
        pred_groups.append(preds[start:end])
        start = end

    # 3) compute and average NDCG@k
    scores = [
        ndcg_score([t], [p], k=k)
        for t, p in zip(true_groups, pred_groups)
    ]
    return float(np.mean(scores))

# ------------------
# SHAP Feature Validation
# ------------------

def shap_feature_validation(
    model,
    X_val: pd.DataFrame,
    features: list
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values on a random subsample of the validation set.
    Supports tree‐based models (via TreeExplainer) and linear models (via LinearExplainer).
    """
    # ── SAMPLE FOR SPEED ───────────────────────────────────────────────
    sample_size = min(1000, X_val.shape[0])
    X_sample = X_val.sample(n=sample_size, random_state=RANDOM_SEED)

    # ── SELECT EXPLAINER ───────────────────────────────────────────────
    from sklearn.linear_model import LinearRegression

    if isinstance(model, (RidgeCV, LinearRegression)):
        explainer = shap.LinearExplainer(model, X_sample, feature_dependence="independent")
        shap_values = explainer.shap_values(X_sample)
    else:
        # handle VotingRegressor or single tree models
        # if it's a VotingRegressor, average SHAP across tree estimators
        try:
            estimators = model.estimators_
        except AttributeError:
            estimators = [model]
        # accumulate
        shap_accum = None
        for est in estimators:
            tree_expl = shap.TreeExplainer(est)
            sv = tree_expl.shap_values(X_sample)
            shap_accum = sv if shap_accum is None else shap_accum + sv
        shap_values = shap_accum / len(estimators)

    # ── AGGREGATE IMPORTANCE ───────────────────────────────────────────
    shap_abs = np.abs(shap_values)
    mean_shap = np.mean(shap_abs, axis=0)
    shap_df = pd.DataFrame({'feature': features, 'mean_abs_shap': mean_shap})
    shap_df.sort_values('mean_abs_shap', ascending=False, inplace=True)

    # ── OUTPUT ────────────────────────────────────────────────────────
    logging.info("\nSHAP Feature Importances (mean |SHAP|) on subsample:")
    logging.info(shap_df.to_string(index=False))
    return shap_df
# ------------------
# Backtest & Evaluation
# ------------------

def backtest_strategy(
    top_trades: pd.DataFrame,
    hold_days: int,
    top_k: int,
    initial_capital: float = 100_000.0
) -> dict:
    """
    Realistic backtester with:
      - Starting cash `initial_capital`
      - Position sizing by normalized size_factor
      - Option contract cost = mid_price * 100
      - Integer contract counts, capital lock‐up until exit
    """
    total_trading_days = top_trades['trade_date'].nunique()
    commission_per_contract = 1.00
    slippage_pct = 0.001  # 0.1%

    ledger = top_trades[['trade_date', 'mid_price', 'size_factor', 'ret_forward']].copy()
    ledger['exit_date'] = ledger['trade_date'] + pd.to_timedelta(hold_days, unit='D')
    trades_by_entry = {d: df for d, df in ledger.groupby('trade_date')}

    capital = initial_capital
    active  = []   # each dict has: exit_date, contracts, cost, ret_forward
    equity_curve = []
    all_days = sorted(set(ledger['trade_date']).union(ledger['exit_date']))

    for today in all_days:
        # 1) Process exits
        for t in active[:]:
            if t['exit_date'] == today:
                # realized PnL per contract
                pnl = t['cost'] * t['ret_forward']
                pnl -= (commission_per_contract + t['cost'] * slippage_pct)
                capital += t['contracts'] * (t['cost'] + pnl)
                active.remove(t)

        # 2) Process new entries
        if today in trades_by_entry:
            df_new = trades_by_entry[today]
            sf = df_new['size_factor'].values
            weights = sf / sf.sum() if sf.sum() > 0 else np.ones(len(sf)) / len(sf)
            daily_budget = initial_capital / total_trading_days

            for (_, row), w in zip(df_new.iterrows(), weights):
                cost = row['mid_price'] * 100.0
                alloc = daily_budget * w
                n_contracts = int(alloc // cost)
                if n_contracts > 0:
                    active.append({
                        'exit_date':   row['exit_date'],
                        'contracts':   n_contracts,
                        'cost':        cost,
                        'ret_forward': row['ret_forward']
                    })
                    capital -= n_contracts * cost

        # 3) Record end‐of‐day equity (free cash + cost‐basis of locked positions)
        locked_cost = sum(t['contracts'] * t['cost'] for t in active)
        equity_curve.append({
            'date':  today,
            'equity': capital + locked_cost
        })

    # Build metrics
    eq_df    = pd.DataFrame(equity_curve).set_index('date').sort_index()
    daily_ret = eq_df['equity'].pct_change().fillna(0)

    total_return       = eq_df['equity'].iloc[-1] / initial_capital - 1
    annualized_sharpe  = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    max_drawdown       = (eq_df['equity'] / eq_df['equity'].cummax() - 1).min()

    metrics = {
        'total_return':      total_return,
        'annualized_sharpe': annualized_sharpe,
        'max_drawdown':      max_drawdown
    }
    logging.info(f"\nBacktest Metrics: {metrics}")
    return metrics

# ------------------
# Date Sanity Check
# ------------------

def data_sanity_checks(df: pd.DataFrame, feature_cols: list, target_col: str = 'ret_forward') -> None:
    logging.info("Running data sanity checks…")
    # 1. Missingness
    missing = df[feature_cols + [target_col]].isnull().mean() * 100
    logging.info("\nMissing values (%) per column:")
    logging.info(missing.to_string())
    # 2. Duplicates
    dups = df.duplicated(subset=['trade_date', 'option_symbol']).sum()
    logging.info(f"\nDuplicate rows (trade_date, option_symbol): {dups}")
    # 3. Target distribution
    logging.info("\nTarget distribution stats:")
    logging.info(df[target_col].describe())
    # 4. Feature descriptive stats
    logging.info("\nFeature descriptive stats:")
    logging.info(df[feature_cols].describe())
    # 5. Feature ↔ target correlations
    corr = df[feature_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
    logging.info("\nFeature ↔ target correlations:")
    logging.info(corr.to_string())

def handle_missing_values(df: pd.DataFrame,
                          feature_cols: list,
                          drop_thresh: float = 0.2) -> pd.DataFrame:
    """
    1. Drop any feature with > drop_thresh fraction of missing values.
    2. Median-impute the remaining missing entries.
    """
    # 1. Compute missing fractions
    missing_frac = df[feature_cols].isnull().mean()
    # 2. Drop high-missing features
    to_drop = missing_frac[missing_frac > drop_thresh].index.tolist()
    if to_drop:
        logging.info(f"Dropping {len(to_drop)} features due to >{drop_thresh*100:.0f}% missing: {to_drop}")
    else:
        logging.info("No features dropped for missingness")
    keep_feats = [f for f in feature_cols if f not in to_drop]
    # 3. Median imputation
    medians = df[keep_feats].median()
    df[keep_feats] = df[keep_feats].fillna(medians)
    return df, keep_feats

# ------------------
# Main Entry Point
# ------------------

def main():
    # ─── Args & Logging ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Option Return Prediction Pipeline')
    parser.add_argument('--hold_days', type=int,   default=HOLD_DAYS_DEFAULT)
    parser.add_argument('--val_frac',  type=float, default=0.2)
    parser.add_argument('--test_frac', type=float, default=0.2)
    parser.add_argument('--top_k',     type=int,   default=TOP_K_DEFAULT)
    parser.add_argument('--retrain', action='store_true',
                        help="Force re-running hyperparameter searches")
    args = parser.parse_args()
    force_retrain = args.retrain
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ─── Load & Prepare ─────────────────────────────────────────────────
    # ─── Load raw features & sweep over candidate hold_days ─────────────────
    df_raw = load_feature_dataframe()
    # ─── Re-build labels & final pipeline on the best hold_days ────────────────
    df = generate_return_label(df_raw, args.hold_days)
    # ─── WINSORIZE TARGET ────────────────────────────────────────────────
    p1, p99 = df['ret_forward'].quantile([0.01, 0.99])
    logging.info(f"Winsorizing ret_forward to [{p1:.4f}, {p99:.4f}]")
    df.loc[:, 'ret_forward'] = df['ret_forward'].clip(lower=p1, upper=p99)

    for feat in ['implied_volatility']:
        if feat in RAW_FEATURE_COLS:
            df.loc[:, feat] = np.log1p(df[feat].clip(lower=0))
            logging.info(f"Clipped & log1p-transformed {feat}")
    
    df_clean = df.dropna(subset=RAW_FEATURE_COLS + ['ret_forward'])

    # ─── Split ──────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_data(df_clean, args.val_frac, args.test_frac)
    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()

    X_parents = train_df[["delta", "moneyness"]].values
    y_inter   = train_df["delta_x_moneyness"].values
    β, *_     = np.linalg.lstsq(X_parents, y_inter, rcond=None)
    np.save("beta_vector.npy", β)
    # 2) Compute the residuals in train & val
    train_df["delta_x_moneyness_resid"] = y_inter - X_parents.dot(β)

    X_parents_val = val_df[["delta", "moneyness"]].values
    y_inter_val   = val_df["delta_x_moneyness"].values
    val_df["delta_x_moneyness_resid"] = y_inter_val - X_parents_val.dot(β)

    # ─── ALSO residualize on the test set ───────────────────────────
    X_parents_test = test_df[["delta", "moneyness"]].values
    y_inter_test   = test_df["delta_x_moneyness"].values
    test_df["delta_x_moneyness_resid"] = y_inter_test - X_parents_test.dot(β)

    FEATURE_COLS = RAW_FEATURE_COLS + ['delta_x_moneyness_resid']

    # Prepare feature & target arrays
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["ret_forward"]
    # per-sample group labels for splitting
    groups_train_labels = train_df["trade_date"].values
    # per-day group sizes for LightGBM ranker
    groups_train_sizes  = train_df.groupby("trade_date").size().tolist()

    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df["ret_forward"]
    groups_val_labels = val_df["trade_date"].values
    groups_val_sizes  = val_df.groupby("trade_date").size().tolist()

    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["ret_forward"]
    groups_test_labels = test_df["trade_date"].values
    groups_test_sizes  = test_df.groupby("trade_date").size().tolist()

    # ─── STEP A: Train base MSE model ───────────────────────────────────
    best_base = tune_best_base_model_params(X_train, y_train, X_val, y_val, force_retrain)
    mean_model = LGBMRegressor(**best_base, random_state=RANDOM_SEED, n_jobs=-1)
    mean_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2',
        callbacks=LGB_EARLY_STOPPING
    )

    # ─── STEP B: Hyper‐tuned quantile models ────────────────────────────
    q_models = train_quantile_models_persisted(
        X_train, y_train,
        X_val,   y_val,
        quantiles=[0.1,0.5,0.9],
        force_retrain=force_retrain
    )
    logging.info("Trained quantile models (α=0.1,0.5,0.9)")

    # We use 5 quantile‐bins (0 through 4) as relevance labels for LambdaRank
    train_df.loc[:, 'relevance'] = pd.qcut(
        train_df['ret_forward'],
        q=5,
        labels=False,
        duplicates='drop'
    ).astype(int)
    val_df.loc[:, 'relevance'] = pd.qcut(
        val_df['ret_forward'],
        q=5,
        labels=False,
        duplicates='drop'
    ).astype(int)
    test_df.loc[:, 'relevance'] = pd.qcut(
        test_df['ret_forward'],
        q=5,
        labels=False,
        duplicates='drop'     # in case of many identical returns
    ).astype(int)
    logging.info("Binned ret_forward into integer `relevance` labels")

    # ─── STEP C: Train LambdaRanker ────────────────────────────────────
    # Use the integer relevance labels for ranking
    y_train_rel = train_df["relevance"]
    y_val_rel   = val_df  ["relevance"]

    ranker = train_lambdarank_persisted(
        X_train, y_train_rel, groups_train_sizes,
        X_val,   y_val_rel,   groups_val_sizes,
        force_retrain
    )
    logging.info("Trained LGBMRanker on integer relevance labels")

    # ─── STEP F: OOF Stacking with GroupKFold ───────────────────────────

    # 1) Prepare OOF container
    n = X_train.shape[0]
    oof_meta = np.zeros((n, 5))   # cols: mean, q10, q50, q90, rank
    gkf = GroupKFold(n_splits=5)

    # 2) Loop folds
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train_labels)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
        gr_tr = train_df.iloc[tr_idx].groupby("trade_date").size().tolist()
        gr_va = train_df.iloc[va_idx].groupby("trade_date").size().tolist()
        y_rel_tr = y_train_rel.iloc[tr_idx]
        y_rel_va = y_train_rel.iloc[va_idx]

        # retrain your base learners on this fold
        m = LGBMRegressor(**best_base, random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va,y_va)], callbacks=LGB_EARLY_STOPPING)
        q10 = LGBMRegressor(**q_models[0.1].get_params()).fit(
                X_tr,y_tr, eval_set=[(X_va,y_va)], callbacks=LGB_EARLY_STOPPING
            ).predict(X_va)
        q50 = LGBMRegressor(**q_models[0.5].get_params()).fit(
                X_tr,y_tr, eval_set=[(X_va,y_va)], callbacks=LGB_EARLY_STOPPING
            ).predict(X_va)
        q90 = LGBMRegressor(**q_models[0.9].get_params()).fit(
                X_tr,y_tr, eval_set=[(X_va,y_va)], callbacks=LGB_EARLY_STOPPING
            ).predict(X_va)
        r_model = clone(ranker)
        r_model.fit(
            X_tr,
            y_rel_tr,
            group=gr_tr,
            eval_set=[(X_va, y_rel_va)],
            eval_group=[gr_va],
            callbacks=LGB_EARLY_STOPPING
        )
        r = r_model.predict(X_va)
        oof_meta[va_idx, :] = np.vstack([
            m.predict(X_va), q10, q50, q90, r
        ]).T

    # 3) Train final Ridge on the full OOF set
    meta = RidgeCV(alphas=[0.1,1.0,10.0], cv=5)
    meta.fit(oof_meta, y_train)

    # 4) Build test meta‐features & evaluate
    test_meta = np.vstack([
        mean_model .predict(X_test),
        q_models[0.1].predict(X_test),
        q_models[0.5].predict(X_test),
        q_models[0.9].predict(X_test),
        ranker    .predict(X_test)
    ]).T

    rmse_stack = np.sqrt(mean_squared_error(y_test,   meta.predict(test_meta)))
    ndcg_stack = evaluate_ndcg(
        meta.predict(test_meta), None,
        test_df['relevance'].values,
        groups_test_sizes
    )
    logging.info(f"OOF-Stack RMSE: {rmse_stack:.4f}, NDCG@5: {ndcg_stack:.4f}")

    preds_stack = meta.predict(test_meta)
    backtest_df = test_df.copy()
    backtest_df['pred_return'] = preds_stack
    backtest_df['size_factor'] = preds_stack

    metrics_stack = backtest_strategy(
        backtest_df,
        hold_days=args.hold_days,           # use the horizon we just selected via CV
        top_k=args.top_k,
        initial_capital=100_000
    )

    result = {
        "hold_days": args.hold_days,
        "rmse":       float(rmse_stack),
        "ndcg":       float(ndcg_stack),
        "sharpe":     float(metrics_stack["annualized_sharpe"])
    }
    print(json.dumps(result))

    # ─── Persist trained models ───────────────────────────────────────
    joblib.dump(mean_model,      "mean_model.pkl")
    joblib.dump(q_models,        "quantile_models.pkl")
    joblib.dump(ranker,          "ranker_model.pkl")
    joblib.dump(meta,            "meta_model.pkl")
    logging.info(
        "Saved models to disk: mean_model.pkl, quantile_models.pkl, "
        "ranker_model.pkl, meta_model.pkl"
    )

if __name__ == '__main__':
    main()
