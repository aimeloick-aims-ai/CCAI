import numpy as np


def create_ridge_stats(dim, n_targets):
    return {
        "n": 0,
        "sum_x": np.zeros(
            dim,
            dtype=np.float64,
        ),
        "sum_x2": np.zeros(
            dim,
            dtype=np.float64,
        ),
        # Full X-X second moment is required for multivariate Ridge.
        "sum_xx": np.zeros(
            (dim, dim),
            dtype=np.float64,
        ),
        "sum_y": np.zeros(
            n_targets,
            dtype=np.float64,
        ),
        "sum_y2": np.zeros(
            n_targets,
            dtype=np.float64,
        ),
        "sum_xy": np.zeros(
            (dim, n_targets),
            dtype=np.float64,
        ),
    }


def update_ridge_stats(stats, X, Y):
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    Y = np.asarray(
        Y,
        dtype=np.float64,
    )

    if Y.ndim == 1:
        Y = Y[:, None]

    n = X.shape[0]

    if n == 0:
        return

    stats["n"] += n

    stats["sum_x"] += X.sum(axis=0)

    stats["sum_x2"] += (
        X * X
    ).sum(axis=0)

    # FIX: accumulate the full X-X second moment.
    stats["sum_xx"] += X.T @ X

    stats["sum_y"] += Y.sum(axis=0)

    stats["sum_y2"] += (
        Y * Y
    ).sum(axis=0)

    stats["sum_xy"] += X.T @ Y


def fit_ridge_from_stats(stats, alpha):
    n = stats["n"]

    if n <= 0:
        raise RuntimeError(
            "No samples available for probe."
        )

    mean_x = stats["sum_x"] / n
    mean_y = stats["sum_y"] / n

    var_x = (
        stats["sum_x2"] / n
        - mean_x * mean_x
    )

    var_x = np.maximum(
        var_x,
        1e-12,
    )

    std_x = np.sqrt(var_x)

    var_y = (
        stats["sum_y2"] / n
        - mean_y * mean_y
    )

    var_y = np.maximum(
        var_y,
        1e-12,
    )

    std_y = np.sqrt(var_y)

    if "sum_xx" not in stats:
        raise RuntimeError(
            "Missing 'sum_xx' in probe statistics. "
            "The Ridge probe requires the full X-X covariance."
        )

    cov_xx = (
        stats["sum_xx"] / n
        - np.outer(mean_x, mean_x)
    )

    cov_xx = 0.5 * (
        cov_xx + cov_xx.T
    )

    cov_xy = (
        stats["sum_xy"] / n
        - np.outer(mean_x, mean_y)
    )

    cov_xx_std = (
        cov_xx
        / std_x[:, None]
        / std_x[None, :]
    )

    cross_std = (
        cov_xy
        / std_x[:, None]
        / std_y[None, :]
    )

    matrix = (
        cov_xx_std
        + alpha * np.eye(len(mean_x))
    )

    coef_std = np.linalg.solve(
        matrix,
        cross_std,
    )

    coef_raw = (
        coef_std
        / std_x[:, None]
    )

    directions = []

    for j in range(
        coef_raw.shape[1]
    ):
        direction = coef_raw[:, j]

        norm = np.linalg.norm(
            direction
        )

        if norm < 1e-12:
            raise RuntimeError(
                "Probe direction has near-zero norm."
            )

        directions.append(
            direction / norm
        )

    directions = np.stack(
        directions,
        axis=0,
    )

    return {
        "mean_x": mean_x,
        "std_x": std_x,
        "mean_y": mean_y,
        "std_y": std_y,
        "coef_std": coef_std,
        "coef_raw": coef_raw,
        "directions": directions,
        "n": n,
    }


def predict_probe(X, probe):
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    X_std = (
        X - probe["mean_x"]
    ) / probe["std_x"]

    pred_std = (
        X_std @ probe["coef_std"]
    )

    pred = (
        pred_std
        * probe["std_y"]
        + probe["mean_y"]
    )

    return pred


def create_prediction_stats(n_targets):
    return {
        "n": 0,
        "sum_y": np.zeros(
            n_targets,
            dtype=np.float64,
        ),
        "sum_y2": np.zeros(
            n_targets,
            dtype=np.float64,
        ),
        "sum_abs_error": np.zeros(
            n_targets,
            dtype=np.float64,
        ),
        "sum_sq_error": np.zeros(
            n_targets,
            dtype=np.float64,
        ),
    }


def update_prediction_stats(
    stats,
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_true.ndim == 1:
        y_true = y_true[:, None]

    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    stats["n"] += len(y_true)

    stats["sum_y"] += (
        y_true.sum(axis=0)
    )

    stats["sum_y2"] += (
        y_true * y_true
    ).sum(axis=0)

    error = y_pred - y_true

    stats["sum_abs_error"] += (
        np.abs(error)
    ).sum(axis=0)

    stats["sum_sq_error"] += (
        error * error
    ).sum(axis=0)


def finalize_prediction_stats(stats):
    n = stats["n"]

    if n <= 0:
        raise RuntimeError(
            "No prediction samples available."
        )

    mse = (
        stats["sum_sq_error"] / n
    )

    mae = (
        stats["sum_abs_error"] / n
    )

    denominator = (
        stats["sum_y2"]
        - stats["sum_y"] ** 2 / n
    )

    denominator = np.maximum(
        denominator,
        1e-12,
    )

    r2 = (
        1.0
        - stats["sum_sq_error"]
        / denominator
    )

    return {
        "r2": r2,
        "mae": mae,
        "mse": mse,
    }
