import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_logistic_regression(X_train, y_train, X_val, y_val, C_values=None):
    """
    Train Logistic Regression models with different C values and select the best based on AP.

    Returns:
        tuple: best_model, validation_predictions, results_df
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    results = []

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]

        results.append({
            "C": C,
            "auc": roc_auc_score(y_val, y_pred),
            "ap": average_precision_score(y_val, y_pred)
        })

    results_df = pd.DataFrame(results).sort_values("ap", ascending=False).reset_index(drop=True)
    logging.info(f"Logistic tuning results (sorted by AP):\n{results_df}")

    best_C = results_df.loc[0, "C"]
    logging.info(f"Selected best C: {best_C}")

    best_model = LogisticRegression(C=best_C, max_iter=1000, class_weight='balanced')
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_val_pred)
    ap = average_precision_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred.round())

    logging.info(f"Validation AUC: {auc:.4f}")
    logging.info(f"Validation Average Precision (AP): {ap:.4f}")
    logging.info(f"Validation F1 Score: {f1:.4f}")

    return best_model, y_val_pred, results_df


def plot_roc_pr(y_true, y_pred, model_name="Logistic Regression", save_path=None):
    """
    Plot ROC and Precision-Recall curves.
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_pred):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'{model_name} (AP = {average_precision_score(y_true, y_pred):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"ROC and PR curves saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    PROJECT_ROOT = Path().resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PREPROCESSING_DIR = DATA_DIR / "preprocessing"

    # Load preprocessed and scaled data
    df_train_scaled = pd.read_pickle(PREPROCESSING_DIR / "X_train_scaled.pkl")
    df_val_scaled = pd.read_pickle(PREPROCESSING_DIR / "X_val_scaled.pkl")
    df_test_scaled = pd.read_pickle(PREPROCESSING_DIR / "X_test_scaled.pkl")
    y_train = pd.read_pickle(PREPROCESSING_DIR / "y_train.pkl")
    y_val = pd.read_pickle(PREPROCESSING_DIR / "y_val.pkl")
    y_test = pd.read_pickle(PREPROCESSING_DIR / "y_test.pkl")

    # Train model
    best_model, y_val_pred, results_df = train_logistic_regression(
        df_train_scaled, y_train, df_val_scaled, y_val
    )

    # Plot ROC and PR curves
    plot_roc_pr(y_val, y_val_pred, model_name="Logistic Regression", save_path="logreg_curves.png")
