import pickle
import logging
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
    Train Logistic Regression models with different C values
    and select the best based on AP.

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

    results_df = (
        pd.DataFrame(results)
        .sort_values("ap", ascending=False)
        .reset_index(drop=True)
    )
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


def plot_confusion_matrix(y_true,
                          y_pred,
                          threshold=0.5,
                          model_name="Model",
                          save_path=None):
    """
    Plot confusion matrix for given true and predicted labels and return fig, ax.
    """
    y_pred_labels = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {model_name} (Threshold={threshold})')

    if save_path:
        fig.savefig(save_path)
        logging.info(f"Confusion matrix saved to {save_path}")

    plt.show()
    return fig, ax


def plot_roc_pr(y_true, y_pred, model_name="Logistic Regression", save_path=None):
    """
    Plot ROC and Precision-Recall curves and return fig, ax.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax[0].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curve')
    ax[0].legend()

    # Precision-Recall Curve
    ax[1].plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logging.info(f"ROC and PR curves saved to {save_path}")
    plt.show()

    return fig, ax


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PREPROCESSING_DIR = DATA_DIR / "preprocessing"
    MODEL_PATH = Path(__file__).resolve().parent / "best_logistic_regression_model.pkl"

    df_train_scaled = pd.read_pickle(PREPROCESSING_DIR / "X_train_scaled.pkl")
    df_val_scaled = pd.read_pickle(PREPROCESSING_DIR / "X_val_scaled.pkl")
    df_test_scaled = pd.read_pickle(PREPROCESSING_DIR / "X_test_scaled.pkl")
    y_train = pd.read_pickle(PREPROCESSING_DIR / "y_train.pkl")
    y_val = pd.read_pickle(PREPROCESSING_DIR / "y_val.pkl")
    y_test = pd.read_pickle(PREPROCESSING_DIR / "y_test.pkl")
    logging.info("Loaded preprocessed and scaled datasets.")

    best_model, y_val_pred, results_df = train_logistic_regression(
        df_train_scaled, y_train, df_val_scaled, y_val
    )
    # Save the best model
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = models_dir / f"best_logistic_regression_model_{timestamp}.pkl"

    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    logging.info(f"Best Logistic Regression model saved to {model_file}")

    fig, ax = plot_roc_pr(
            y_val,
            y_val_pred,
            model_name="Logistic Regression",
            save_path=Path(__file__).resolve().parent / "logreg_curves.png")
    plt.show()

    fig, ax = plot_confusion_matrix(
            y_val,
            y_val_pred,
            threshold=0.5,
            model_name="Logistic Regression",
            save_path=Path(__file__).resolve().parent / "logreg_confusion_matrix.png")
    plt.show()

    # Evaluate on Test Set
    y_test_pred = best_model.predict_proba(df_test_scaled)[:, 1]

    test_auc = roc_auc_score(y_test, y_test_pred)
    test_ap = average_precision_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred.round())

    logging.info(f"Test AUC: {test_auc:.4f}")
    logging.info(f"Test Average Precision (AP): {test_ap:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")

    fig, ax = plot_roc_pr(
        y_test, y_test_pred,
        model_name="Logistic Regression - Test",
        save_path=Path(__file__).resolve().parent / "logreg_test_curves.png"
    )

    fig, ax = plot_confusion_matrix(
        y_test, y_test_pred,
        threshold=0.5,
        model_name="Logistic Regression - Test",
        save_path=Path(__file__).resolve().parent / "logreg_test_confusion_matrix.png"
    )
