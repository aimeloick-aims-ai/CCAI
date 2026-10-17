import numpy as np


NUM_CLASSES = 7


def create_confusion():
    return np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )


def update_confusion(
    confusion,
    y_true,
    y_pred,
):
    y_true = (
        y_true.detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        y_pred.detach()
        .cpu()
        .numpy()
    )

    for t, p in zip(
        y_true,
        y_pred,
    ):
        confusion[
            int(t),
            int(p),
        ] += 1


def confusion_metrics(confusion):
    total = confusion.sum()

    accuracy = (
        np.trace(confusion) / total
        if total > 0
        else 0.0
    )

    f1_values = []

    for c in range(NUM_CLASSES):
        tp = confusion[c, c]

        fp = (
            confusion[:, c].sum()
            - tp
        )

        fn = (
            confusion[c, :].sum()
            - tp
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        f1_values.append(f1)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(
            np.mean(f1_values)
        ),
        "f1_per_class": np.asarray(
            f1_values
        ),
    }
