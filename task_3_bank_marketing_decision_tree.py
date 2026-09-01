"""Task 3: Decision Tree Classifier for Bank Marketing prediction.

This script loads the Bank Marketing dataset from the Prodigy-InfoTech sample
repository, preprocesses the categorical features, trains a decision tree model,
and evaluates whether a customer is likely to subscribe to a term deposit.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

DATASET_URL = (
    "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/"
    "main/Task%203/bank/bank.csv"
)


def load_data():
    """Load the Bank Marketing dataset from the GitHub source."""
    return pd.read_csv(DATASET_URL, sep=';')


def prepare_data(df):
    """Prepare features and target for modeling."""
    df_clean = df.copy()

    # Convert target to binary numeric values.
    target_map = {"yes": 1, "no": 0}
    df_clean["y"] = df_clean["y"].map(target_map)

    X = df_clean.drop(columns=["y"])
    y = df_clean["y"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    return X, y, preprocessor


def train_model(X, y, preprocessor):
    """Train and return a decision tree classifier pipeline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=6,
                    min_samples_leaf=10,
                    criterion="gini",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Model Evaluation ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("task3_outputs/confusion_matrix.png")
    plt.close()

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    print("\nTop 10 Most Important Features:\n")
    print(feature_importance_df.head(10).to_string(index=False))

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(10), x="importance", y="feature", palette="viridis")
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("task3_outputs/feature_importance.png")
    plt.close()

    return pipeline


def show_dataset_summary(df):
    """Print a quick overview of the dataset."""
    print("\n=== Dataset Overview ===")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nTarget distribution:\n", df["y"].value_counts())
    print("\nMissing values:\n", df.isnull().sum())


def main():
    output_dir = Path("task3_outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_data()
    show_dataset_summary(df)

    X, y, preprocessor = prepare_data(df)
    train_model(X, y, preprocessor)

    print("\nAll outputs were saved in the 'task3_outputs' folder.")


if __name__ == "__main__":
    main()
