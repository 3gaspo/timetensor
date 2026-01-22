import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform, pdist, cosine, cdist
from sklearn.manifold import TSNE
import ipywidgets as widgets
from IPython.display import display, clear_output


# --------- utils ---------

def set_seed(seed):
    """Sets RNG seeds when seed is not None."""
    if seed == "None":
        seed = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


def symlog(x, linthresh=1):
    """Signed log transform with linear threshold."""
    return np.sign(x) * np.log1p(np.abs(x / linthresh)) * linthresh


def normalize(x, mean, std, eps=1e-8):
    """Normalizes x using mean/std with epsilon."""
    return (x - mean) / (std + eps)


def filter_df(df, mask):
    """Masks df entries where mask is True."""
    df = df.copy()
    df[mask] = pd.NA
    return df


def filter_dict(dico, keys):
    """Filters a dict by a list of keys."""
    return {key: dico[key] for key in keys}


def cte_mask(df, lookback):
    """Returns mask of constant rolling windows of length lookback."""
    stds = df.rolling(window=lookback).std()
    return stds == 0


def get_normal_stats(x):
    """Returns per-sample mean/std over last dimension.
    x: (B, dim, dates)
    means: (B, dim, 1)
    """
    mean = x.mean(dim=-1, keepdim=True).detach()
    std = x.std(dim=-1, keepdim=True).detach()
    return mean, std


def unroll_windows(dataloader, cap=None, normal=False, mean=None, std=None, seed=None):
    """Unrolls windows from a torch dataloader into tensors."""
    set_seed(seed)

    X, Y, C = [], [], []
    carry_on, total = True, 0
    while carry_on:
        for x, c, y, indiv, date in dataloader:
            total += x.shape[0]
            if normal:
                if mean is None and std is None:
                    mean, std = get_normal_stats(x)
                x, y = normalize(x, mean, std), normalize(y, mean, std)
            X.append(x)
            Y.append(y)
            C.append(c)
            if cap is not None and total + x.shape[0] > cap:
                carry_on = True
                break
        if cap is None or total >= cap:
            carry_on = False

    return torch.concat(X), torch.concat(Y), torch.concat(C)


def get_trend(df, window=1000):
    """Rolling mean trend."""
    return df.rolling(window=window).mean().iloc[window:]


def get_aggr(df, window=100):
    """Block-wise mean aggregation with block size window."""
    n = len(df)
    if n == 0:
        return df.copy()
    block_ids = np.arange(n) // window
    block_means = df.groupby(block_ids).mean()
    aggr_df = df.copy()
    for pos, idx in enumerate(df.index):
        aggr_df.loc[idx] = block_means.loc[block_ids[pos]]
    return aggr_df


def split_six_way(df, time_splits=(0.6, 0.4), indiv_split=1.0, seed=0):
    """Six-way split for dataframe."""
    set_seed(seed)

    if len(time_splits) not in (2, 3):
        raise ValueError("time_splits must have length 2 or 3")

    n = len(df)
    cols = list(df.columns)

    if len(time_splits) == 2:
        a, b = time_splits
        t1 = int(a * n)
        t2 = n
    else:
        a, b, c = time_splits
        t1 = int(a * n)
        t2 = int((a + b) * n)

    k = len(cols)
    k_primary = int(indiv_split * k)
    perm = np.random.permutation(cols)
    primary_cols = list(perm[:k_primary])
    secondary_cols = list(perm[k_primary:])

    df_primary = df[primary_cols] if primary_cols else df.iloc[:, :0]
    df_secondary = df[secondary_cols] if secondary_cols else df.iloc[:, :0]

    if len(time_splits) == 2:
        train = df_primary.iloc[:t1]
        valid1 = df_primary.iloc[:0]
        test1 = df_primary.iloc[t1:]

        valid2 = df_secondary.iloc[:t1]
        valid3 = df_secondary.iloc[:0]
        test2 = df_secondary.iloc[t1:]
    else:
        train = df_primary.iloc[:t1]
        valid1 = df_primary.iloc[t1:t2]
        test1 = df_primary.iloc[t2:]

        valid2 = df_secondary.iloc[:t1]
        valid3 = df_secondary.iloc[t1:t2]
        test2 = df_secondary.iloc[t2:]

    return {
        "train": train,
        "valid1": valid1,
        "test1": test1,
        "valid2": valid2,
        "valid3": valid3,
        "test2": test2,
    }


# --------- stats core ---------

def get_fourier_df(df, eps=1e-8):
    """Per-column FFT magnitude of standardized series."""
    return df.apply(lambda x: np.abs(np.fft.fft((x - x.mean()) / (x.std() + eps))))


def get_gammas(data, lookback, horizon, eps=1e-8):
    """Returns alpha/beta dataframes from rolling lookback/horizon stats."""
    lookback_means = data.rolling(window=lookback).mean().iloc[lookback:]
    lookback_stds = data.rolling(window=lookback).std().iloc[lookback:]
    horizon_means = data.rolling(window=horizon).mean().shift(-horizon).iloc[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon).iloc[:-horizon]
    alphas = horizon_stds.iloc[lookback:] / (lookback_stds.iloc[:-horizon] + eps)
    betas = (horizon_means.iloc[lookback:] - lookback_means.iloc[:-horizon]) / (lookback_stds.iloc[:-horizon] + eps)
    return alphas, betas


def get_gamma_df(df, lags, horizon, eps=1e-8):
    """Concatenates alpha and beta into a single dataframe."""
    alphas_df, betas_df = get_gammas(df, lags, horizon, eps=eps)
    gamma_df = pd.concat((alphas_df, betas_df))
    return gamma_df


def get_dataset_stats(df_dict, lags, horizon, sampling, save_path=None):
    """Computes dataset-wide mean/std and average alpha/beta for each split."""
    gammas_dict = {k: get_gammas(df_dict[k], lags, horizon) for k in df_dict}
    stats_dict = {}
    for key in df_dict:
        if (key == "train" and sampling["remove_train_cte"]) or (key != "train" and sampling["remove_eval_cte"]):
            mask = cte_mask(df_dict[key], lags)
            clean_df = filter_df(df_dict[key], mask)
            clean_alphas = filter_df(gammas_dict[key][0], mask)
            clean_betas = filter_df(gammas_dict[key][1], mask)
        else:
            clean_df, clean_alphas, clean_betas = df_dict[key], gammas_dict[key][0], gammas_dict[key][1]
        stats_dict[key] = {
            "mean": float(np.nanmean(clean_df.values)),
            "stds": float(np.nanmean(np.nanstd(clean_df.values, axis=0))),
            "std": float(np.nanstd(clean_df.values)),
            "alpha": float(np.nanmean(clean_alphas)),
            "beta": float(np.nanmean(clean_betas)),
        }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(stats_dict, f, indent=4)

    return stats_dict


# --------- heterogeneities ---------

def energy_distance_multivariate(X1, X2):
    """Energy distance between two multivariate samples."""
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    d_xx = cdist(X1, X1)
    d_yy = cdist(X2, X2)
    d_xy = cdist(X1, X2)
    return np.sqrt(max(2 * d_xy.mean() - d_xx.mean() - d_yy.mean(), 0))


def identify_cte(df, lookback, show=True, save_path=""):
    """Finds and optionally plots constant rolling window counts."""
    stds_mask = cte_mask(df, lookback)
    row_idxs, col_idxs = np.where(stds_mask)

    counts = {}
    for j in col_idxs:
        counts[j] = counts.get(j, 0) + 1

    total = int(len(col_idxs))

    if counts:
        vals = np.array(list(counts.values()))
        max_col_idx = max(counts, key=counts.get)
        print(f"Found {len(counts)} users with constant windows!")
        print(f"Total windows: {total}")
        print(f"Max per user:  {counts[max_col_idx]} (user {df.columns[max_col_idx]})")
        print(f"Mean per user: {vals.mean():.2f}")

        plt.figure(figsize=(6, 4))
        plt.hist(list(counts.values()), bins=100)
        plt.yscale("log")
        plt.title("Constant windows per individual")
        plt.xlabel("Individuals")
        plt.ylabel("log(counts)")

        if show:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + "constants_hist.pdf")
        plt.close()
    else:
        print("No constant windows found!")


# --------- window sampling + window stats ---------

def sample_windows_df(df, lookback, horizon, n_windows, columns=None, ignore_cte=False, seed=None):
    """Samples windows from df and returns lookbacks and horizons arrays."""
    set_seed(seed)

    if columns is None:
        columns = list(df.columns)
    else:
        columns = [c for c in columns if c in df.columns]

    X, Y = [], []
    L, H, N = int(lookback), int(horizon), int(n_windows)

    if L <= 0 or H <= 0 or N <= 0 or len(columns) == 0:
        return np.empty((0, L)), np.empty((0, H))

    for col in columns:
        x = df[col].values
        n = len(x)
        if n < L + H:
            continue

        possible_t = np.arange(L, n - H + 1)
        if len(possible_t) == 0:
            continue

        needed = N - len(X)
        if needed <= 0:
            break

        if needed >= len(possible_t):
            t_indices = possible_t
        else:
            t_indices = np.random.choice(possible_t, size=needed, replace=False)

        for t in t_indices:
            look = x[t - L:t]
            if ignore_cte and np.std(look) == 0:
                continue
            X.append(look)
            Y.append(x[t:t + H])

        if len(X) >= N:
            break

    if len(X) == 0:
        return np.empty((0, L)), np.empty((0, H))

    X = np.asarray(X)[:N]
    Y = np.asarray(Y)[:N]
    return X, Y


def window_mean_std(windows):
    """Computes per-window mean/std from lookbacks."""
    x = np.asarray(windows)
    if x.size == 0:
        return np.array([]), np.array([])
    return x.mean(axis=1), x.std(axis=1)


def window_alpha_beta(lookbacks, horizons, eps=1e-6):
    """Computes per-window alpha/beta from lookbacks and horizons."""
    X = np.asarray(lookbacks)
    Y = np.asarray(horizons)
    if X.size == 0 or Y.size == 0:
        return np.array([]), np.array([])

    mL = X.mean(axis=1)
    sL = X.std(axis=1)
    mH = Y.mean(axis=1)
    sH = Y.std(axis=1)

    denom = sL + eps
    alpha = sH / denom
    beta = (mH - mL) / denom
    return alpha, beta


# --------- widgets: series ---------

def plot_series_widget(
    df,
    trend_window=1000,
    aggr_window=100,
    gamma_lookback=168,
    gamma_horizon=24,
    gamma_eps=1e-8,
):
    """Interactive series widget caching only current-parameter dataframes."""
    cache = {"dfs": {}, "params": {}}

    dataframe_names = ["original", "alpha", "beta", "means", "aggregate"]
    column_names = list(df.columns)

    dataframe_dropdown = widgets.Dropdown(options=dataframe_names, value="original", description="Data:")
    column_dropdown = widgets.Dropdown(options=column_names, value=column_names[0], description="User:")
    next_button = widgets.Button(description="Next")
    axis_button = widgets.Button(description="Toggle Axis")
    apply_button = widgets.Button(description="Apply Params")

    L_widget = widgets.IntText(value=gamma_lookback, description="L:")
    H_widget = widgets.IntText(value=gamma_horizon, description="H:")
    trend_widget = widgets.IntText(value=trend_window, description="Trend:")
    aggr_widget = widgets.IntText(value=aggr_window, description="Aggr:")

    output = widgets.Output()
    axis_state = {"show": True}

    def params_from_ui():
        return {
            "gamma_lookback": int(L_widget.value),
            "gamma_horizon": int(H_widget.value),
            "trend_window": int(trend_widget.value),
            "aggr_window": int(aggr_widget.value),
            "gamma_eps": float(gamma_eps),
        }

    def relevant_widgets(name):
        if name in ("alpha", "beta"):
            return [L_widget, H_widget]
        if name == "means":
            return [trend_widget]
        if name == "aggregate":
            return [aggr_widget]
        return []

    def update_param_ui():
        name = dataframe_dropdown.value
        for w in [L_widget, H_widget, trend_widget, aggr_widget]:
            w.layout.display = "none"
        for w in relevant_widgets(name):
            w.layout.display = "block"

    def compute_current_all():
        p = params_from_ui()
        cache["params"] = p
        cache["dfs"]["original"] = df
        alphas_df, betas_df = get_gammas(df, p["gamma_lookback"], p["gamma_horizon"], eps=p["gamma_eps"])
        cache["dfs"]["alpha"] = alphas_df
        cache["dfs"]["beta"] = betas_df
        cache["dfs"]["means"] = get_trend(df, window=p["trend_window"])
        cache["dfs"]["aggregate"] = get_aggr(df, window=p["aggr_window"])

    def recompute_selected():
        p = params_from_ui()
        cache["params"] = p
        name = dataframe_dropdown.value
        if name in ("alpha", "beta"):
            alphas_df, betas_df = get_gammas(df, p["gamma_lookback"], p["gamma_horizon"], eps=p["gamma_eps"])
            cache["dfs"]["alpha"] = alphas_df
            cache["dfs"]["beta"] = betas_df
        elif name == "means":
            cache["dfs"]["means"] = get_trend(df, window=p["trend_window"])
        elif name == "aggregate":
            cache["dfs"]["aggregate"] = get_aggr(df, window=p["aggr_window"])
        elif name == "original":
            cache["dfs"]["original"] = df

    def update_plot():
        with output:
            output.clear_output(wait=True)
            name = dataframe_dropdown.value
            col = column_dropdown.value
            if name not in cache["dfs"]:
                compute_current_all()
            df_current = cache["dfs"][name]
            if col not in df_current.columns:
                print(f"Column '{col}' not in DataFrame '{name}'.")
                return
            plt.figure(figsize=(15, 4))
            plt.plot(df_current[col])
            if axis_state["show"]:
                plt.title(f"{name} - {col}")
                plt.grid(True)
                plt.xlabel("Index")
                plt.ylabel("Value")
            else:
                plt.axis("off")
            plt.show()

    def on_apply(b):
        recompute_selected()
        update_plot()

    def on_dropdown_change(change):
        update_param_ui()
        update_plot()

    def on_next_button_click(b):
        idx = column_names.index(column_dropdown.value)
        column_dropdown.value = column_names[(idx + 1) % len(column_names)]

    def on_axis_button_click(b):
        axis_state["show"] = not axis_state["show"]
        update_plot()

    compute_current_all()
    update_param_ui()

    dataframe_dropdown.observe(on_dropdown_change, names="value")
    column_dropdown.observe(on_dropdown_change, names="value")
    next_button.on_click(on_next_button_click)
    axis_button.on_click(on_axis_button_click)
    apply_button.on_click(on_apply)

    display(
        widgets.HBox([dataframe_dropdown, column_dropdown, next_button, axis_button]),
        widgets.HBox([L_widget, H_widget, trend_widget, aggr_widget, apply_button]),
        output,
    )
    update_plot()


def plot_window_widget(df, default_lookback=168, default_horizon=24, eps=1e-6):
    """Interactive single-window visualization widget."""
    full_data = df.copy()
    columns = list(full_data.columns)
    n_rows = full_data.shape[0]

    t_widget = widgets.IntText(value=n_rows // 2, description='t:')
    L_widget = widgets.IntText(value=default_lookback, description='L:')
    H_widget = widgets.IntText(value=default_horizon, description='H:')
    user_dropdown = widgets.Dropdown(options=columns, value=columns[0], description='User:')
    norm_button = widgets.ToggleButton(value=False, description='Normalize')
    output_plot = widgets.Output()
    output_stats = widgets.Output()

    def update(change):
        with output_plot:
            output_plot.clear_output(wait=True)

            t = t_widget.value
            L = int(L_widget.value)
            H = int(H_widget.value)
            col = user_dropdown.value
            do_norm = bool(norm_button.value)

            if L <= 0:
                print("L must be > 0.")
                return
            if H <= 0:
                print("H must be > 0.")
                return

            if t < L:
                t_widget.value = L
                return
            if t + H > n_rows:
                t_widget.value = n_rows - H
                return

            series = full_data[col].values
            look = series[t - L:t]
            hor = series[t:t + H]

            m = float(np.mean(look))
            s = float(np.std(look))

            alpha, beta = window_alpha_beta(look[None, :], hor[None, :], eps=eps)
            alpha = float(alpha[0]) if alpha.size else None
            beta = float(beta[0]) if beta.size else None

            lookback_vals = full_data[col].iloc[t - L:t].copy()
            horizon_vals = full_data[col].iloc[t:t + H].copy()

            if do_norm and s > 0:
                lookback_vals = (lookback_vals - m) / s
                horizon_vals = (horizon_vals - m) / s
                ylabel = "Z-score"
                title_suffix = " (normalized)"
            else:
                ylabel = "Value"
                title_suffix = ""

            plt.figure(figsize=(12, 4))
            plt.plot(range(t - L, t), lookback_vals, color='blue', label=f'L={L}')
            plt.plot(range(t, t + H), horizon_vals, color='orange', label=f'H={H}')
            plt.axvline(x=t, color='gray', linestyle='--', label='split')
            plt.title(f'{col} - t={t}{title_suffix}')
            plt.xlabel('Index')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            plt.show()

        with output_stats:
            output_stats.clear_output(wait=True)
            if alpha is not None:
                print("--- Stats ---")
                print(f"mean:  {m:.4f}")
                print(f"std:   {s:.4f}")
                print(f"alpha: {alpha:.4f}")
                print(f"beta:  {beta:.4f}")
            else:
                print("Invalid window.")

    t_widget.observe(update, names='value')
    L_widget.observe(update, names='value')
    H_widget.observe(update, names='value')
    user_dropdown.observe(update, names='value')
    norm_button.observe(update, names='value')

    display(widgets.HBox([t_widget, L_widget, H_widget, user_dropdown, norm_button]))
    display(widgets.HBox([output_plot, output_stats]))
    update(None)


# --------- widgets: stats ---------

def plot_stats_widget(df, seed=None):
    """Interactive stats widget caching only current sampled points."""
    full_data = df.copy()
    columns = list(full_data.columns)
    cache = {"params": None, "plot_df": {"Mean/Std": None, "Alpha/Beta": None}}

    L_widget = widgets.IntText(value=168, description="L:")
    H_widget = widgets.IntText(value=24, description="H:")
    N_widget = widgets.IntText(value=100, description="N:")
    user_dropdown = widgets.Dropdown(options=columns, value=columns[0], description="User:")
    type_dropdown = widgets.Dropdown(options=["Mean/Std", "Alpha/Beta"], value="Mean/Std", description="Type:")
    log_button = widgets.ToggleButton(value=True, description="Log")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")
    resample_button = widgets.Button(description="Resample")
    next_button = widgets.Button(description="Next User")
    output = widgets.Output()

    def compute_cache(L, H, N, ignore_cte):
        set_seed(seed)
        records_ms = []
        records_ab = []
        linthresh = 1

        for col in columns:
            X, Y = sample_windows_df(full_data, L, H, N, columns=[col], ignore_cte=ignore_cte, seed=seed)

            means, stds = window_mean_std(X)
            for m, s in zip(means, stds):
                records_ms.append({"user": col, "mean": m, "std": s})

            alphas, betas = window_alpha_beta(X, Y)
            for a, b in zip(alphas, betas):
                records_ab.append({"user": col, "alpha": a, "beta": b})

        df_ms = pd.DataFrame(records_ms) if records_ms else None
        df_ab = pd.DataFrame(records_ab) if records_ab else None

        if df_ms is not None and not df_ms.empty:
            df_ms["mean_symlog"] = symlog(df_ms["mean"], linthresh=linthresh)
            df_ms["std_symlog"] = symlog(df_ms["std"], linthresh=linthresh)

        if df_ab is not None and not df_ab.empty:
            df_ab["alpha_symlog"] = symlog(df_ab["alpha"], linthresh=linthresh)
            df_ab["beta_symlog"] = symlog(df_ab["beta"], linthresh=linthresh)

        cache["plot_df"]["Mean/Std"] = df_ms
        cache["plot_df"]["Alpha/Beta"] = df_ab
        cache["params"] = (L, H, N, ignore_cte)

    def ensure_cached():
        L = int(L_widget.value)
        H = int(H_widget.value)
        N = int(N_widget.value)
        ignore_cte = bool(filter_button.value)
        params = (L, H, N, ignore_cte)
        if cache["params"] != params:
            compute_cache(L, H, N, ignore_cte)

    def plot_current():
        with output:
            output.clear_output(wait=True)

            ensure_cached()

            plot_type = type_dropdown.value
            plot_df = cache["plot_df"].get(plot_type, None)
            if plot_df is None or plot_df.empty:
                print("No data.")
                return

            highlight = user_dropdown.value
            use_log = bool(log_button.value)

            plot_df = plot_df.copy()
            plot_df["type"] = plot_df["user"].apply(lambda u: "user" if u == highlight else "all")
            plot_df = plot_df.sort_values(by="type", ascending=True)

            if plot_type == "Mean/Std":
                x_base, y_base = "mean", "std"
                x_label_raw, y_label_raw = "Mean", "Std"
            else:
                x_base, y_base = "beta", "alpha"
                x_label_raw, y_label_raw = "Beta", "Alpha"

            if use_log:
                x_col = f"{x_base}_symlog"
                y_col = f"{y_base}_symlog"
                x_label = f"{x_label_raw} (log)"
                y_label = f"{y_label_raw} (log)"
            else:
                x_col, y_col = x_base, y_base
                x_label, y_label = x_label_raw, y_label_raw

            plt.figure(figsize=(6, 5))
            g = sns.jointplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                hue="type",
                palette={"user": "red", "all": "blue"},
                kind="scatter",
                height=7,
                s=20,
                marginal_kws=dict(common_norm=False, fill=True, alpha=0.5),
            )

            highlight_df = plot_df[plot_df["type"] == "user"]
            g.ax_joint.scatter(highlight_df[x_col], highlight_df[y_col], color="red", s=20, alpha=1, zorder=10)
            g.ax_joint.set_xlabel(x_label)
            g.ax_joint.set_ylabel(y_label)
            g.figure.suptitle(f"L={int(L_widget.value)}, H={int(H_widget.value)}, N={int(N_widget.value)} - {plot_type}", y=1.02)
            plt.show()

    def on_resample(b):
        cache["params"] = None
        plot_current()

    def on_next(b):
        idx = columns.index(user_dropdown.value)
        user_dropdown.value = columns[(idx + 1) % len(columns)]

    for w in [L_widget, H_widget, N_widget, user_dropdown, type_dropdown, log_button, filter_button]:
        w.observe(lambda change: plot_current(), names="value")

    resample_button.on_click(on_resample)
    next_button.on_click(on_next)

    ensure_cached()
    display(
        widgets.HBox([L_widget, H_widget, N_widget]),
        widgets.HBox([type_dropdown, log_button, filter_button]),
        widgets.HBox([user_dropdown, next_button, resample_button]),
        output,
    )
    plot_current()


def plot_stats_dict_widget(df_dict, seed=None):
    """Window stats widget over multiple datasets caching only current sampled points."""
    keys = list(df_dict.keys())
    cache = {"params": None, "plot_df": {"Mean/Std": None, "Alpha/Beta": None}}

    L_widget = widgets.IntText(value=168, description="L:")
    H_widget = widgets.IntText(value=24, description="H:")
    N_widget = widgets.IntText(value=100, description="N (ref):")
    type_dropdown = widgets.Dropdown(options=["Mean/Std", "Alpha/Beta"], value="Mean/Std", description="Type:")
    log_button = widgets.ToggleButton(value=True, description="Log")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")
    resample_button = widgets.Button(description="Resample")
    output = widgets.Output()

    def compute_cache(L, H, N_ref, ignore_cte):
        set_seed(seed)
        records_ms = []
        records_ab = []
        linthresh = 1

        ref_len = len(df_dict[keys[0]])
        if ref_len == 0:
            cache["plot_df"]["Mean/Std"] = None
            cache["plot_df"]["Alpha/Beta"] = None
            cache["params"] = (L, H, N_ref, ignore_cte)
            return

        for name in keys:
            df = df_dict[name]
            n_dates = len(df)
            if n_dates == 0:
                continue

            N = max(1, int(n_dates / float(ref_len) * N_ref))
            for col in list(df.columns):
                X, Y = sample_windows_df(df, L, H, N, columns=[col], ignore_cte=ignore_cte, seed=seed)

                means, stds = window_mean_std(X)
                for m, s in zip(means, stds):
                    records_ms.append({"dataset": name, "mean": m, "std": s})

                alphas, betas = window_alpha_beta(X, Y)
                for a, b in zip(alphas, betas):
                    records_ab.append({"dataset": name, "alpha": a, "beta": b})

        df_ms = pd.DataFrame(records_ms) if records_ms else None
        df_ab = pd.DataFrame(records_ab) if records_ab else None

        if df_ms is not None and not df_ms.empty:
            df_ms["mean_symlog"] = symlog(df_ms["mean"], linthresh=linthresh)
            df_ms["std_symlog"] = symlog(df_ms["std"], linthresh=linthresh)

        if df_ab is not None and not df_ab.empty:
            df_ab["alpha_symlog"] = symlog(df_ab["alpha"], linthresh=linthresh)
            df_ab["beta_symlog"] = symlog(df_ab["beta"], linthresh=linthresh)

        cache["plot_df"]["Mean/Std"] = df_ms
        cache["plot_df"]["Alpha/Beta"] = df_ab
        cache["params"] = (L, H, N_ref, ignore_cte)

    def ensure_cached():
        L = int(L_widget.value)
        H = int(H_widget.value)
        N_ref = int(N_widget.value)
        ignore_cte = bool(filter_button.value)
        params = (L, H, N_ref, ignore_cte)
        if cache["params"] != params:
            compute_cache(L, H, N_ref, ignore_cte)

    def plot_current():
        with output:
            output.clear_output(wait=True)

            ensure_cached()

            plot_type = type_dropdown.value
            plot_df = cache["plot_df"].get(plot_type, None)
            if plot_df is None or plot_df.empty:
                print("No data.")
                return

            use_log = bool(log_button.value)

            if plot_type == "Mean/Std":
                x_base, y_base = "mean", "std"
                x_label_raw, y_label_raw = "Mean", "Std"
            else:
                x_base, y_base = "beta", "alpha"
                x_label_raw, y_label_raw = "Beta", "Alpha"

            if use_log:
                x_col = f"{x_base}_symlog"
                y_col = f"{y_base}_symlog"
                x_label = f"{x_label_raw} (log)"
                y_label = f"{y_label_raw} (log)"
            else:
                x_col, y_col = x_base, y_base
                x_label, y_label = x_label_raw, y_label_raw

            plt.figure(figsize=(6,5))
            g = sns.jointplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                hue="dataset",
                kind="scatter",
                height=7,
                s=20,
                marginal_kws=dict(common_norm=False, fill=True, alpha=0.5),
            )

            g.ax_joint.set_xlabel(x_label)
            g.ax_joint.set_ylabel(y_label)
            g.fig.suptitle(
                f"L={int(L_widget.value)}, H={int(H_widget.value)}, N_ref={int(N_widget.value)} - {plot_type}",
                y=1.02
            )
            plt.show()

    def on_resample(b):
        cache["params"] = None
        plot_current()

    for w in [L_widget, H_widget, N_widget, type_dropdown, log_button, filter_button]:
        w.observe(lambda change: plot_current(), names="value")

    resample_button.on_click(on_resample)

    ensure_cached()
    display(
        widgets.HBox([L_widget, H_widget, N_widget]),
        widgets.HBox([type_dropdown, log_button, filter_button, resample_button]),
        output,
    )
    plot_current()


def plot_distances_dict_widget(df_dict, seed=None):
    """Distance-matrix widget over datasets caching current-parameter distance matrices for all normalizations."""
    assert "train" in df_dict, "df_dict must include a 'train' dataframe for standard normalization."

    keys = list(df_dict.keys())
    cache = {"params": None, "matrices": None, "train_stats": None}

    L_widget = widgets.IntText(value=168, description="L:")
    H_widget = widgets.IntText(value=24, description="H:")
    N_widget = widgets.IntText(value=100, description="N (ref):")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")
    norm_dropdown = widgets.Dropdown(
        options=["raw", "standard", "instance"],
        value="raw",
        description="Norm:",
    )
    dist_dropdown = widgets.Dropdown(
        options=["raw input", "raw joint", "mean/std", "alpha/beta"],
        value="raw input",
        description="Dist:",
    )
    apply_button = widgets.Button(description="Apply")
    output = widgets.Output()

    def get_train_stats():
        if cache["train_stats"] is not None:
            return cache["train_stats"]
        train_vals = df_dict["train"].values.astype(float)
        mu = float(np.nanmean(train_vals))
        sig = float(np.nanstd(train_vals))
        cache["train_stats"] = (mu, sig)
        return cache["train_stats"]

    def standard_normalize(X, Y, eps=1e-8):
        if X.size == 0 or Y.size == 0:
            return X, Y
        mu, sig = get_train_stats()
        denom = sig + eps
        return (X - mu) / denom, (Y - mu) / denom

    def instance_normalize(X, Y, eps=1e-8):
        if X.size == 0 or Y.size == 0:
            return X, Y
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True)
        mask = (s.squeeze(-1) > 0)
        Xn = X.copy()
        Yn = Y.copy()
        Xn[mask] = (Xn[mask] - m[mask]) / (s[mask] + eps)
        Yn[mask] = (Yn[mask] - m[mask]) / (s[mask] + eps)
        return Xn, Yn

    def apply_norm(X, Y, norm_mode):
        if norm_mode == "raw":
            return X, Y
        if norm_mode == "standard":
            return standard_normalize(X, Y)
        if norm_mode == "instance":
            return instance_normalize(X, Y)
        raise ValueError("Unknown normalization mode.")

    def compute_matrices_for_norm(L, H, N_ref, ignore_cte, norm_mode):
        set_seed(seed)
        ref_len = len(df_dict[keys[0]])
        if ref_len == 0:
            return None

        samples = {}
        feats = {}

        for name in keys:
            df = df_dict[name]
            n_dates = len(df)
            N = max(1, int(n_dates / float(ref_len) * N_ref)) if ref_len > 0 else max(1, N_ref)

            X, Y = sample_windows_df(df, L, H, N, columns=None, ignore_cte=ignore_cte, seed=seed)
            X, Y = apply_norm(X, Y, norm_mode)
            J = np.concatenate([X, Y], axis=1) if (X.size and Y.size) else np.empty((0, L + H))

            samples[name] = {"X": X, "J": J}

            m, s = window_mean_std(X)
            a, b = window_alpha_beta(X, Y)
            feats[name] = {
                "mean/std": np.stack([m, s], axis=1) if len(m) else np.empty((0, 2)),
                "alpha/beta": np.stack([a, b], axis=1) if len(a) else np.empty((0, 2)),
            }

        def pairwise_matrix(mode):
            M = np.full((len(keys), len(keys)), np.nan, dtype=float)
            for i, ki in enumerate(keys):
                for j, kj in enumerate(keys):
                    if i == j:
                        M[i, j] = 0.0
                        continue

                    if mode == "raw input":
                        A, B = samples[ki]["X"], samples[kj]["X"]
                    elif mode == "raw joint":
                        A, B = samples[ki]["J"], samples[kj]["J"]
                    elif mode == "mean/std":
                        A, B = feats[ki]["mean/std"], feats[kj]["mean/std"]
                    elif mode == "alpha/beta":
                        A, B = feats[ki]["alpha/beta"], feats[kj]["alpha/beta"]
                    else:
                        raise ValueError("Unknown mode")

                    if A.size == 0 or B.size == 0:
                        M[i, j] = np.nan
                    else:
                        M[i, j] = energy_distance_multivariate(A, B)
            return M

        return {
            "raw input": pairwise_matrix("raw input"),
            "raw joint": pairwise_matrix("raw joint"),
            "mean/std": pairwise_matrix("mean/std"),
            "alpha/beta": pairwise_matrix("alpha/beta"),
        }

    def compute_all_norms(L, H, N_ref, ignore_cte):
        get_train_stats()
        mats = {}
        for norm_mode in ["raw", "standard", "instance"]:
            mats[norm_mode] = compute_matrices_for_norm(L, H, N_ref, ignore_cte, norm_mode)
        return mats

    def ensure_cached(force=False):
        L = int(L_widget.value)
        H = int(H_widget.value)
        N_ref = int(N_widget.value)
        ignore_cte = bool(filter_button.value)
        params = (L, H, N_ref, ignore_cte)
        if force or cache["params"] != params:
            cache["matrices"] = compute_all_norms(L, H, N_ref, ignore_cte)
            cache["params"] = params

    def plot_current():
        with output:
            output.clear_output(wait=True)
            ensure_cached()

            mats_all = cache["matrices"]
            if mats_all is None:
                print("No data.")
                return

            norm_mode = norm_dropdown.value
            mats = mats_all.get(norm_mode, None)
            if mats is None:
                print("No data.")
                return

            mode = dist_dropdown.value
            M = mats[mode]

            plt.figure(figsize=(4 + 0.35 * len(keys), 3 + 0.25 * len(keys)))
            im = plt.imshow(M, aspect="auto")
            plt.colorbar(im)

            plt.xticks(np.arange(len(keys)), keys, rotation=45, ha="right")
            plt.yticks(np.arange(len(keys)), keys)

            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
                    plt.text(j, i, txt, ha="center", va="center")

            plt.title(
                f"Distances ({mode}) | Norm={norm_mode} | L={int(L_widget.value)}, H={int(H_widget.value)}, N_ref={int(N_widget.value)}"
            )
            plt.tight_layout()
            plt.show()

    def on_apply(b):
        ensure_cached(force=True)
        plot_current()

    def on_force_recompute(change):
        ensure_cached(force=True)
        plot_current()

    dist_dropdown.observe(lambda change: plot_current(), names="value")
    norm_dropdown.observe(lambda change: plot_current(), names="value")
    filter_button.observe(on_force_recompute, names="value")
    apply_button.on_click(on_apply)

    ensure_cached(force=True)
    display(
        widgets.HBox([L_widget, H_widget, N_widget]),
        widgets.HBox([filter_button, norm_dropdown, dist_dropdown, apply_button]),
        output,
    )
    plot_current()


# --------- clustering core ---------

def calculate_distances(df, metric='cosine', matrix=False):
    """Pairwise distances between columns."""
    D = pdist(df.T.values, metric=metric)
    if matrix:
        return squareform(D)
    return D


def find_pairs(distances_matrix):
    """Returns closest and furthest users."""
    size = distances_matrix.shape
    na, ma = np.unravel_index(np.argmin(distances_matrix + np.identity(size[0]), axis=None), size)
    nb, mb = np.unravel_index(np.argmax(distances_matrix, axis=None), size)
    return na, nb, ma, mb


def init_clusters(df):
    """Initializes hierarchical clustering linkage and distances matrix."""
    distances = calculate_distances(df)
    Z = shc.linkage(distances, method='average')
    return Z, squareform(distances)


def get_clusters(Z, n_clusters):
    """Computes flat clusters from linkage."""
    labels = shc.fcluster(Z, n_clusters, criterion='maxclust')
    cluster_indices = [np.where(labels == i)[0] for i in range(1, n_clusters + 1)]
    return labels, cluster_indices


def get_centroids(df, cluster_indices):
    """Computes per-cluster centroids."""
    centroids = []
    for indices in cluster_indices:
        cluster_data = df.iloc[:, indices]
        centroids.append(cluster_data.mean(axis=1))
    return centroids


def get_cluster_dicts(df, cluster_indices):
    """Builds dict of cluster->sub-dataframe."""
    clusters = {}
    for i, indices in enumerate(cluster_indices):
        clusters[f'cluster_{i}'] = df.iloc[:, indices]
    return clusters


def get_cluster_distances(df, cluster_indices):
    """Computes intra- and inter-cluster cosine distances."""
    intra_distances, inter_distances = {}, {}
    centroids = get_centroids(df, cluster_indices)

    for i, idx in enumerate(cluster_indices):
        if len(idx) > 1:
            d = []
            for j in range(len(idx)):
                for k in range(j + 1, len(idx)):
                    d.append(cosine(df.iloc[:, idx[j]].values, df.iloc[:, idx[k]].values))
            intra_distances[i] = np.mean(d)
        else:
            intra_distances[i] = np.nan

        if len(cluster_indices) > 1:
            for j in range(i + 1, len(cluster_indices)):
                inter_distances[(i, j)] = cosine(centroids[i].values, centroids[j].values)
        else:
            inter_distances = {0: 0}

    return intra_distances, inter_distances


def get_cluster_heterogeneity(df, cluster_indices):
    """Returns heterogeneity proxy from intra/inter distances."""
    intra_distances, inter_distances = get_cluster_distances(df, cluster_indices)
    intra = list(intra_distances.values())
    inter = list(inter_distances.values())
    if len(inter) > 0:
        return np.nanmean(intra) / (np.mean(inter) + 1)
    else:
        return np.nanmean(intra)


def plot_distances(distances_matrix, show=True, path="", name="distances.pdf"):
    """Plots histogram of distances."""
    plt.figure(figsize=(10, 4))
    plt.hist(distances_matrix[np.triu_indices(distances_matrix.shape[0], k=1)], bins=100)
    plt.title("Distances histogram")
    plt.xlabel("Distances")
    plt.ylabel("Counts")
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()


def plot_dendogram(Z, show=True, path="", name="dendogram.pdf"):
    """Plots dendrogram."""
    plt.figure(figsize=(15, 4))
    shc.dendrogram(Z)
    plt.title("Dendogram")
    plt.xticks([])
    plt.xlabel("")
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()


def plot_clusters(df, cluster_indices, n_examples, show=False, path=""):
    """Plots example series per cluster."""
    for i, indices in enumerate(cluster_indices):
        print(f"Cluster {i + 1}:")
        for j in range(min(n_examples, len(indices))):
            sample_index = indices[j]
            print(f"  Sample index: {sample_index}")
            plt.figure(figsize=(20, 3))
            plt.plot(df.iloc[:, sample_index], c=f"C{i}")
            plt.title(f'Sample {sample_index} from cluster {i + 1}')
            if show:
                plt.show()
            else:
                plt.savefig(path + f"cluster{i + 1}_id{sample_index}.pdf")
            plt.close()


def plot_centroids(centroids, show=True, path="", name="centroids.pdf"):
    """Plots cluster centroids."""
    plt.figure(figsize=(15, 4))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f'Cluster {i + 1}')
    plt.title('Centroids of clusters')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()


def plot_heterogeneity(df, show=True, path="", name="heterogeneity.pdf", N_clusters=None, seed=None):
    """Plots heterogeneity vs number of clusters."""
    set_seed(seed)

    heterogeneities = []
    if N_clusters is None:
        N_clusters = [1, 2, 3, 4, 5, 10, 20, df.shape[1] // 10, df.shape[1] // 5, df.shape[1] // 2, df.shape[1] // 1, df.shape[1]]
    N_clusters = np.sort(N_clusters)

    Z, _ = init_clusters(df)

    for n_clusters in tqdm(N_clusters):
        _, cluster_indices = get_clusters(Z, n_clusters)
        heterogeneities.append(get_cluster_heterogeneity(df, cluster_indices))

    plt.figure(figsize=(6, 4))
    plt.plot(N_clusters, heterogeneities)
    plt.xlabel("Number of clusters")
    plt.ylabel("Heterogeneity")
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()


# --------- widgets: clustering ---------

def plot_centroids_widget(df):
    """Interactive centroid plotting widget with current-only cluster indices."""
    cache = {
        "dfs": {},
        "gamma_params": None,
        "cluster_indices": None,
        "labels": None,
        "cluster_params": None,
    }

    dataset_dropdown = widgets.Dropdown(options=['raw', 'fourier', 'gamma'], value='fourier', description='Data:')
    n_clusters_slider = widgets.IntSlider(min=2, max=min(30, df.shape[1]), step=1, value=3, description='Clusters:', continuous_update=False)
    lags_slider = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=168, description='Lags:')
    horizon_slider = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=24, description='Horizon:')
    output = widgets.Output()

    def compute_dfs():
        cache["dfs"]["raw"] = df
        cache["dfs"]["fourier"] = get_fourier_df(df)
        cache["dfs"]["gamma"] = get_gamma_df(df, lags=int(lags_slider.value), horizon=int(horizon_slider.value))
        cache["gamma_params"] = (int(lags_slider.value), int(horizon_slider.value))

    def ensure_gamma_df():
        gp = (int(lags_slider.value), int(horizon_slider.value))
        if cache["gamma_params"] != gp:
            cache["dfs"]["gamma"] = get_gamma_df(df, lags=gp[0], horizon=gp[1])
            cache["gamma_params"] = gp

    def toggle_gamma(change):
        val = change.get('new', None) if not isinstance(change, str) else change
        if val == 'gamma':
            lags_slider.layout.display = 'block'
            horizon_slider.layout.display = 'block'
        else:
            lags_slider.layout.display = 'none'
            horizon_slider.layout.display = 'none'

    def update(change=None):
        with output:
            clear_output(wait=True)

            dataset_type = dataset_dropdown.value
            n_clusters = int(n_clusters_slider.value)

            if dataset_type == "gamma":
                ensure_gamma_df()

            current_df = cache["dfs"][dataset_type]
            max_clusters = current_df.shape[1]
            if n_clusters > max_clusters:
                n_clusters = max_clusters

            Z, _ = init_clusters(current_df)
            labels, cluster_indices = get_clusters(Z, n_clusters)

            cache["labels"] = labels
            cache["cluster_indices"] = cluster_indices
            cache["cluster_params"] = (dataset_type, n_clusters, int(lags_slider.value), int(horizon_slider.value))

            print("Feature-based centroids:")
            centroids = get_centroids(current_df, cluster_indices)
            plot_centroids(centroids)

            if dataset_type != 'raw':
                print("Raw centroids:")
                raw_centroids = get_centroids(df, cluster_indices)
                plot_centroids(raw_centroids)

    compute_dfs()
    dataset_dropdown.observe(toggle_gamma, names='value')
    toggle_gamma(dataset_dropdown.value)

    for w in [dataset_dropdown, n_clusters_slider, lags_slider, horizon_slider]:
        w.observe(update, names='value')

    ui = widgets.VBox([dataset_dropdown, n_clusters_slider, lags_slider, horizon_slider])
    display(ui, output)
    update(None)


def plot_clustering_widget(df):
    """Interactive clustering diagnostics widget caching current Z for each dataset type."""
    cache = {
        "dfs": {},
        "gamma_params": None,
        "Z": {"raw": None, "fourier": None, "gamma": None},
        "dist": {"raw": None, "fourier": None, "gamma": None},
    }

    dataset_dropdown = widgets.Dropdown(options=['raw', 'fourier', 'gamma'], value='fourier', description='Data:')
    lags_slider = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=168, description='Lags:', layout=widgets.Layout(display='none'))
    horizon_slider = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=24, description='Horizon:', layout=widgets.Layout(display='none'))
    output = widgets.Output()

    def compute_dfs():
        cache["dfs"]["raw"] = df
        cache["dfs"]["fourier"] = get_fourier_df(df)
        cache["dfs"]["gamma"] = get_gamma_df(df, lags=int(lags_slider.value), horizon=int(horizon_slider.value))
        cache["gamma_params"] = (int(lags_slider.value), int(horizon_slider.value))

    def ensure_gamma_df():
        gp = (int(lags_slider.value), int(horizon_slider.value))
        if cache["gamma_params"] != gp:
            cache["dfs"]["gamma"] = get_gamma_df(df, lags=gp[0], horizon=gp[1])
            cache["gamma_params"] = gp

    def compute_Z_all_initial():
        Z, D = init_clusters(cache["dfs"]["raw"])
        cache["Z"]["raw"], cache["dist"]["raw"] = Z, D
        Z, D = init_clusters(cache["dfs"]["fourier"])
        cache["Z"]["fourier"], cache["dist"]["fourier"] = Z, D
        Z, D = init_clusters(cache["dfs"]["gamma"])
        cache["Z"]["gamma"], cache["dist"]["gamma"] = Z, D

    def update_current_Z_if_needed(dataset_type):
        if dataset_type != "gamma":
            if cache["Z"][dataset_type] is None:
                Z, D = init_clusters(cache["dfs"][dataset_type])
                cache["Z"][dataset_type], cache["dist"][dataset_type] = Z, D
            return

        ensure_gamma_df()
        Z, D = init_clusters(cache["dfs"]["gamma"])
        cache["Z"]["gamma"], cache["dist"]["gamma"] = Z, D

    def update(change=None):
        with output:
            clear_output(wait=True)
            dataset_type = dataset_dropdown.value
            update_current_Z_if_needed(dataset_type)
            Z = cache["Z"][dataset_type]
            distances_matrix = cache["dist"][dataset_type]
            print(f"Dendrogram ({dataset_type}):")
            plot_dendogram(Z)
            print(f"Distances ({dataset_type}):")
            plot_distances(distances_matrix)

    def toggle_gamma(change):
        val = change.get('new', None) if not isinstance(change, str) else change
        if val == 'gamma':
            lags_slider.layout.display = 'block'
            horizon_slider.layout.display = 'block'
        else:
            lags_slider.layout.display = 'none'
            horizon_slider.layout.display = 'none'

    compute_dfs()
    compute_Z_all_initial()

    dataset_dropdown.observe(toggle_gamma, names='value')
    toggle_gamma(dataset_dropdown.value)

    for w in [dataset_dropdown, lags_slider, horizon_slider]:
        w.observe(update, names='value')

    ui = widgets.VBox([dataset_dropdown, lags_slider, horizon_slider])
    display(ui, output)
    update(None)


def plots_stats_clusters_widget(df, seed=None):
    """Joint stats widget colored by current clusters (no param-keyed caching)."""
    cache = {
        "dfs": {},
        "gamma_params": None,
        "stats_params": None,
        "stats_df": None,
        "cluster_params": None,
        "labels": None,
        "plot_df": None,
    }

    columns = list(df.columns)

    dataset_dropdown = widgets.Dropdown(options=['raw', 'fourier', 'gamma'], value='fourier', description='Data:')
    n_clusters_slider = widgets.IntSlider(min=2, max=min(30, df.shape[1]), step=1, value=3, description='Clusters:', continuous_update=False)
    lags_slider = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=168, description='Lags:')
    horizon_slider = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=24, description='Horizon:')

    L_widget = widgets.IntText(value=168, description='L:')
    H_widget = widgets.IntText(value=24, description='H:')
    N_widget = widgets.IntText(value=100, description='N:')
    type_dropdown = widgets.Dropdown(options=['Mean/Std', 'Alpha/Beta'], value='Mean/Std', description='Type:')
    log_button = widgets.ToggleButton(value=True, description='Log')
    filter_button = widgets.ToggleButton(value=False, description='Filter cte')

    output = widgets.Output()

    def compute_dfs():
        cache["dfs"]["raw"] = df
        cache["dfs"]["fourier"] = get_fourier_df(df)
        cache["dfs"]["gamma"] = get_gamma_df(df, lags=int(lags_slider.value), horizon=int(horizon_slider.value))
        cache["gamma_params"] = (int(lags_slider.value), int(horizon_slider.value))

    def ensure_gamma_df():
        gp = (int(lags_slider.value), int(horizon_slider.value))
        if cache["gamma_params"] != gp:
            cache["dfs"]["gamma"] = get_gamma_df(df, lags=gp[0], horizon=gp[1])
            cache["gamma_params"] = gp

    def precompute_stats(L, H, N, plot_type, ignore_cte):
        set_seed(seed)
        records = []
        linthresh = 1

        for col in columns:
            X, Y = sample_windows_df(df, L, H, N, columns=[col], ignore_cte=ignore_cte, seed=seed)

            if plot_type == 'Mean/Std':
                x_vals, y_vals = window_mean_std(X)
                x_name, y_name = 'mean', 'std'
            else:
                x_vals, y_vals = window_alpha_beta(X, Y)
                x_name, y_name = 'alpha', 'beta'

            for xv, yv in zip(x_vals, y_vals):
                records.append({'user': col, x_name: xv, y_name: yv})

        if not records:
            cache["stats_df"] = None
            return

        df_temp = pd.DataFrame(records)
        if plot_type == 'Mean/Std':
            df_temp['mean_symlog'] = symlog(df_temp['mean'], linthresh=linthresh)
            df_temp['std_symlog'] = symlog(df_temp['std'], linthresh=linthresh)
        else:
            df_temp['alpha_symlog'] = symlog(df_temp['alpha'], linthresh=linthresh)
            df_temp['beta_symlog'] = symlog(df_temp['beta'], linthresh=linthresh)

        cache["stats_df"] = df_temp

    def compute_current_clusters(dataset_type, n_clusters):
        if dataset_type == "gamma":
            ensure_gamma_df()

        current_df = cache["dfs"][dataset_type]
        max_clusters = current_df.shape[1]
        if n_clusters > max_clusters:
            n_clusters = max_clusters

        Z, _ = init_clusters(current_df)
        labels, _ = get_clusters(Z, n_clusters)
        cache["labels"] = labels

    def build_plot_df(dataset_type, n_clusters, plot_type):
        stats_df = cache["stats_df"]
        if stats_df is None:
            cache["plot_df"] = None
            return

        compute_current_clusters(dataset_type, n_clusters)

        labels = cache["labels"]
        col_list = list(df.columns)
        label_map = {col_list[i]: labels[i] for i in range(len(col_list))}

        plot_df = stats_df.copy()
        plot_df['cluster'] = plot_df['user'].map(lambda u: f'c{label_map.get(u, 0)}')
        cache["plot_df"] = plot_df

    def update(change=None):
        with output:
            output.clear_output(wait=True)

            dataset_type = dataset_dropdown.value
            n_clusters = int(n_clusters_slider.value)
            L = int(L_widget.value)
            H = int(H_widget.value)
            N = int(N_widget.value)
            plot_type = type_dropdown.value
            use_log = bool(log_button.value)
            ignore_cte = bool(filter_button.value)

            if L <= 0 or N <= 0 or (plot_type == 'Alpha/Beta' and H <= 0):
                print("Invalid parameters.")
                return

            if L > df.shape[0]:
                print("L too large.")
                return
            if plot_type == 'Alpha/Beta' and L + H > df.shape[0]:
                print("L+H too large.")
                return

            stats_params = (L, H, N, plot_type, ignore_cte)
            if cache["stats_params"] != stats_params:
                cache["stats_params"] = stats_params
                precompute_stats(L, H, N, plot_type, ignore_cte)

            cluster_params = (dataset_type, n_clusters, int(lags_slider.value), int(horizon_slider.value), plot_type)
            if cache["cluster_params"] != cluster_params or cache["plot_df"] is None:
                cache["cluster_params"] = cluster_params
                build_plot_df(dataset_type, n_clusters, plot_type)

            plot_df = cache["plot_df"]
            if plot_df is None or plot_df.empty:
                print("No data.")
                return

            if plot_type == 'Mean/Std':
                x_base, y_base = 'mean', 'std'
                x_label_raw, y_label_raw = 'Mean', 'Std'
            else:
                x_base, y_base = 'beta', 'alpha'
                x_label_raw, y_label_raw = 'Beta', 'Alpha'

            if use_log:
                x_col = f'{x_base}_symlog'
                y_col = f'{y_base}_symlog'
                x_label = f'{x_label_raw} (log)'
                y_label = f'{y_label_raw} (log)'
            else:
                x_col, y_col = x_base, y_base
                x_label, y_label = x_label_raw, y_label_raw

            plt.figure(figsize=(6,5))
            g = sns.jointplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                hue='cluster',
                kind='scatter',
                height=7,
                s=20,
                marginal_kws=dict(common_norm=False, fill=True, alpha=0.5),
            )

            g.ax_joint.set_xlabel(x_label)
            g.ax_joint.set_ylabel(y_label)
            g.fig.suptitle(
                f'{dataset_type} | clusters={n_clusters} | L={L}, H={H}, N={N} - {plot_type}',
                y=1.02
            )
            plt.show()

    def toggle_gamma(change):
        val = change.get('new', None) if not isinstance(change, str) else change
        if val == 'gamma':
            lags_slider.layout.display = 'block'
            horizon_slider.layout.display = 'block'
        else:
            lags_slider.layout.display = 'none'
            horizon_slider.layout.display = 'none'

    compute_dfs()

    dataset_dropdown.observe(toggle_gamma, names='value')
    toggle_gamma(dataset_dropdown.value)

    widgets_list = [
        dataset_dropdown, n_clusters_slider, lags_slider, horizon_slider,
        L_widget, H_widget, N_widget, type_dropdown, log_button, filter_button
    ]
    for w in widgets_list:
        w.observe(update, names='value')

    controls_top = widgets.HBox([dataset_dropdown, n_clusters_slider, lags_slider, horizon_slider])
    controls_mid = widgets.HBox([L_widget, H_widget, N_widget])
    controls_bottom = widgets.HBox([type_dropdown, log_button, filter_button])
    display(widgets.VBox([controls_top, controls_mid, controls_bottom]), output)
    update(None)


# --------- widgets: t-SNE (distributions) ---------

def plot_tsne_dict_widget(df_dict, seed=None):
    """
    t-SNE widget over multiple datasets:
      - samples windows from each df in df_dict (proportional to length vs first key)
      - builds per-window vectors from either:
          * inputs only: X (shape N x L)
          * joint: [X, Y] (shape N x (L+H))
      - supports normalization strategies: raw, standard (train stats), instance (per-window)
      - IMPORTANT: t-SNE is computed on the concatenated dataset of all samples, then plotted with different
        colors per df; first df has higher alpha.

    Caching:
      - caches embeddings for current (L, H, N_ref, ignore_cte, mode) and *precomputes all three* norms
        ("raw", "standard", "instance") for those params.
    """
      # local import so script doesn't hard-require sklearn unless used

    assert "train" in df_dict, "df_dict must include a 'train' dataframe for standard normalization."

    keys = list(df_dict.keys())
    first_key = keys[0]

    cache = {
        "params": None,          # (L, H, N_ref, ignore_cte, mode)
        "train_stats": None,     # (mu, sig)
        "embeddings": None,      # dict[norm_mode] -> dict with {"Z": (N,2), "labels": (N,), "sizes": {k:nk}}
    }

    # ---- UI ----
    L_widget = widgets.IntText(value=168, description="L:")
    H_widget = widgets.IntText(value=24, description="H:")
    N_widget = widgets.IntText(value=200, description="N (ref):")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")

    mode_dropdown = widgets.Dropdown(
        options=[("inputs only", "inputs"), ("inputs + outputs (joint)", "joint")],
        value="inputs",
        description="Data:",
    )
    norm_dropdown = widgets.Dropdown(
        options=["raw", "standard", "instance"],
        value="raw",
        description="Norm:",
    )

    # modest defaults that usually work; you can tweak if you want
    perplexity_widget = widgets.IntText(value=30, description="Perp:")
    lr_widget = widgets.IntText(value=200, description="LR:")
    apply_button = widgets.Button(description="Apply")
    output = widgets.Output()

    # ---- helpers ----
    def get_train_stats():
        if cache["train_stats"] is not None:
            return cache["train_stats"]
        train_vals = df_dict["train"].values.astype(float)
        mu = float(np.nanmean(train_vals))
        sig = float(np.nanstd(train_vals))
        cache["train_stats"] = (mu, sig)
        return cache["train_stats"]

    def standard_normalize_XY(X, Y, eps=1e-8):
        if X.size == 0:
            return X, Y
        mu, sig = get_train_stats()
        denom = sig + eps
        Xn = (X - mu) / denom
        Yn = (Y - mu) / denom if (Y is not None and Y.size) else Y
        return Xn, Yn

    def instance_normalize_XY(X, Y, eps=1e-8):
        if X.size == 0:
            return X, Y
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True)
        mask = (s.squeeze(-1) > 0)
        Xn = X.copy()
        Xn[mask] = (Xn[mask] - m[mask]) / (s[mask] + eps)
        Yn2 = None
        if Y is not None and Y.size:
            Yn2 = Y.copy()
            Yn2[mask] = (Yn2[mask] - m[mask]) / (s[mask] + eps)
        else:
            Yn2 = Y
        return Xn, Yn2

    def apply_norm(X, Y, norm_mode):
        if norm_mode == "raw":
            return X, Y
        if norm_mode == "standard":
            return standard_normalize_XY(X, Y)
        if norm_mode == "instance":
            return instance_normalize_XY(X, Y)
        raise ValueError("Unknown normalization mode.")

    def build_concat_samples(L, H, N_ref, ignore_cte, mode, norm_mode):
        """
        Returns:
          A: (N_total, D) concatenated features across datasets
          labels: (N_total,) dataset label per row
          sizes: dict(dataset -> count)
        """
        set_seed(seed)

        ref_len = len(df_dict[first_key])
        if ref_len == 0:
            return np.empty((0, 0)), np.array([]), {}

        feats_all = []
        labels_all = []
        sizes = {}

        for name in keys:
            df = df_dict[name]
            n_dates = len(df)
            if n_dates == 0:
                sizes[name] = 0
                continue

            N = max(1, int(n_dates / float(ref_len) * N_ref)) if ref_len > 0 else max(1, N_ref)

            X, Y = sample_windows_df(df, L, H, N, columns=None, ignore_cte=ignore_cte, seed=seed)
            if X.size == 0:
                sizes[name] = 0
                continue

            X, Y = apply_norm(X, Y, norm_mode)

            if mode == "inputs":
                A = X
            else:
                # joint
                if Y is None or Y.size == 0:
                    A = np.empty((0, L + H))
                else:
                    A = np.concatenate([X, Y], axis=1)

            if A.size == 0:
                sizes[name] = 0
                continue

            feats_all.append(A)
            labels_all.append(np.array([name] * A.shape[0], dtype=object))
            sizes[name] = int(A.shape[0])

        if not feats_all:
            return np.empty((0, 0)), np.array([]), {}

        A = np.concatenate(feats_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)
        return A, labels, sizes

    def compute_tsne_for_norm(A, perplexity, lr):
        """
        Computes 2D embedding.
        Note: TSNE is stochastic; we set random_state from seed for reproducibility.
        """
        if A.size == 0 or A.shape[0] < 2:
            return np.empty((0, 2))
        # keep perplexity valid
        perp = int(perplexity)
        perp = max(2, min(perp, (A.shape[0] - 1) // 3 if A.shape[0] > 6 else 2))
        lr = float(lr)

        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=lr,
            init="pca",
            random_state=None if seed is None else int(seed),
            max_iter=1000,
            verbose=0,
        )
        return tsne.fit_transform(A)

    def ensure_cached(force=False):
        L = int(L_widget.value)
        H = int(H_widget.value)
        N_ref = int(N_widget.value)
        ignore_cte = bool(filter_button.value)
        mode = mode_dropdown.value
        params = (L, H, N_ref, ignore_cte, mode)

        if force or cache["params"] != params:
            # (re)compute embeddings for ALL norms for these params
            get_train_stats()

            perplexity = int(perplexity_widget.value)
            lr = int(lr_widget.value)

            embs = {}
            for norm_mode in ["raw", "standard", "instance"]:
                A, labels, sizes = build_concat_samples(L, H, N_ref, ignore_cte, mode, norm_mode)
                Z = compute_tsne_for_norm(A, perplexity=perplexity, lr=lr)
                embs[norm_mode] = {"Z": Z, "labels": labels, "sizes": sizes}

            cache["embeddings"] = embs
            cache["params"] = params

    def plot_current():
        with output:
            output.clear_output(wait=True)

            ensure_cached(force=False)

            embs = cache["embeddings"]
            if embs is None:
                print("No data.")
                return

            norm_mode = norm_dropdown.value
            pack = embs.get(norm_mode, None)
            if pack is None:
                print("No data.")
                return

            Z = pack["Z"]
            labels = pack["labels"]
            sizes = pack["sizes"]

            if Z.size == 0 or labels.size == 0:
                print("No samples (check L/H/N_ref, or data too short).")
                return

            # colors: one per dataset key (C0, C1, ...)
            key_to_color = {k: f"C{i}" for i, k in enumerate(keys)}

            plt.figure(figsize=(7, 6))

            # plot first df (higher alpha) then others
            for k in keys:
                mask = (labels == k)
                if not np.any(mask):
                    continue
                alpha = 0.85 if k == first_key else 0.25
                plt.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    s=14,
                    alpha=alpha,
                    c=key_to_color[k],
                    label=f"{k} (n={sizes.get(k, 0)})",
                    edgecolors="none",
                )

            L = int(L_widget.value)
            H = int(H_widget.value)
            N_ref = int(N_widget.value)
            ignore_cte = bool(filter_button.value)
            mode = mode_dropdown.value
            perp = int(perplexity_widget.value)
            lr = int(lr_widget.value)

            plt.title(f"t-SNE | mode={mode} | norm={norm_mode} | L={L}, H={H}, N_ref={N_ref}, filter_cte={ignore_cte} | perp={perp}, lr={lr}")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.grid(True, alpha=0.2)
            plt.legend(loc="best", frameon=True)
            plt.tight_layout()
            plt.show()

    def on_apply(_):
        ensure_cached(force=True)
        plot_current()

    def on_simple_change(_):
        # norm switch: just replot; other changes trigger recompute via apply or auto below
        plot_current()

    # auto-update on most controls (like your other widgets)
    for w in [norm_dropdown]:
        w.observe(on_simple_change, names="value")

    # when core params change, we can either auto-recompute (can be heavy) or require Apply.
    # Here we mimic your "Apply" pattern: just redraw message until Apply pressed.
    def on_core_param_change(_):
        with output:
            output.clear_output(wait=True)
            print("Params changed — press Apply to recompute t-SNE.")
    for w in [L_widget, H_widget, N_widget, filter_button, mode_dropdown, perplexity_widget, lr_widget]:
        w.observe(on_core_param_change, names="value")

    apply_button.on_click(on_apply)

    # initial compute + display
    ensure_cached(force=True)
    display(
        widgets.HBox([L_widget, H_widget, N_widget, filter_button]),
        widgets.HBox([mode_dropdown, norm_dropdown, perplexity_widget, lr_widget, apply_button]),
        output,
    )
    plot_current()
