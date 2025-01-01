import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def confusion_matrix_plot(
    y_true, y_pred, labels=None, normalize=False, title=None, cmap="Blues", ax=None
):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : array-like, optional
        List of class labels.
    normalize : bool, default=False
        If True, display normalized results.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """

    # Clear current figure in cache if it exists
    plt.clf()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create annotations with both absolute numbers and percentages
    annotations = np.empty_like(cm, dtype="<U32")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                annotations[i, j] = f"{cm_norm[i,j]:.1%}"
            else:
                annotations[i, j] = f"{cm[i,j]} ({cm_norm[i,j]:.1%})"

    if labels is not None:
        df_cm = pd.DataFrame(
            cm if not normalize else cm_norm, index=labels, columns=labels
        )
    else:
        df_cm = pd.DataFrame(cm if not normalize else cm_norm)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sns.heatmap(df_cm, annot=annotations, fmt="", cmap=cmap, ax=ax)
    ax.invert_yaxis()  # Invert y axis to match matrix representation
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    if title:
        ax.set_title(title)

    return fig
