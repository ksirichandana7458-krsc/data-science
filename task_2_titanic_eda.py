"""Task 2: Data cleaning and exploratory data analysis on the Titanic dataset.

This script loads the Titanic training dataset from the Prodigy-InfoTech sample
repository, cleans missing values, and explores relationships between variables.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_URL = (
    "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/"
    "main/Task%202/train.csv"
)


def load_data():
    """Load the Titanic dataset from the GitHub source."""
    return pd.read_csv(DATASET_URL)


def clean_data(df):
    """Clean missing values and create a few derived features for analysis."""
    df_clean = df.copy()

    # Fill missing values with sensible defaults
    df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
    df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])
    df_clean["Fare"] = df_clean["Fare"].fillna(df_clean["Fare"].median())
    df_clean["Cabin"] = df_clean["Cabin"].fillna("Unknown")

    # Normalize strings
    df_clean["Sex"] = df_clean["Sex"].str.title()
    df_clean["Embarked"] = df_clean["Embarked"].str.title()

    # Derived features
    df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
    df_clean["IsAlone"] = np.where(df_clean["FamilySize"] == 1, 1, 0)

    return df_clean


def print_summary(df):
    """Print dataset summary information."""
    print("\n=== Dataset Overview ===")
    print(df.head())
    print("\n=== Dataset Shape ===")
    print(df.shape)
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== Duplicate Rows ===")
    print(df.duplicated().sum())


def plot_survival_distribution(df, output_dir):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Survived", palette="Set2")
    plt.title("Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "survival_count.png")
    plt.close()


def plot_gender_survival(df, output_dir):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Sex", y="Survived", estimator=np.mean, palette="pastel")
    plt.title("Survival Rate by Gender")
    plt.ylabel("Survival Rate")
    plt.xlabel("Gender")
    plt.tight_layout()
    plt.savefig(output_dir / "gender_survival.png")
    plt.close()


def plot_age_distribution(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, kde=True, palette="Set1")
    plt.title("Age Distribution by Survival Status")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png")
    plt.close()


def plot_class_survival(df, output_dir):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Pclass", hue="Survived", palette="viridis")
    plt.title("Survival by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    plt.legend(title="Survived")
    plt.tight_layout()
    plt.savefig(output_dir / "class_survival.png")
    plt.close()


def plot_embarkation_survival(df, output_dir):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Embarked", hue="Survived", palette="deep")
    plt.title("Survival by Embarkation Port")
    plt.xlabel("Embarked")
    plt.ylabel("Count")
    plt.legend(title="Survived")
    plt.tight_layout()
    plt.savefig(output_dir / "embarked_survival.png")
    plt.close()


def plot_fare_boxplot(df, output_dir):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Survived", y="Fare", palette="coolwarm")
    plt.title("Fare Distribution by Survival Status")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(output_dir / "fare_boxplot.png")
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()


def show_key_insights(df):
    """Display a few observed trends from the cleaned dataset."""
    print("\n=== Key Insights ===")
    print("Overall survival rate:", round(df["Survived"].mean() * 100, 2), "%")
    print("Survival rate by sex:")
    print(df.groupby("Sex")["Survived"].mean().round(3) * 100)
    print("\nSurvival rate by passenger class:")
    print(df.groupby("Pclass")["Survived"].mean().round(3) * 100)
    print("\nAverage age by survival status:")
    print(df.groupby("Survived")["Age"].mean().round(2))
    print("\nAverage fare by survival status:")
    print(df.groupby("Survived")["Fare"].mean().round(2))


def main():
    output_dir = Path("task2_outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_data()
    df_clean = clean_data(df)

    print_summary(df_clean)
    show_key_insights(df_clean)

    plot_survival_distribution(df_clean, output_dir)
    plot_gender_survival(df_clean, output_dir)
    plot_age_distribution(df_clean, output_dir)
    plot_class_survival(df_clean, output_dir)
    plot_embarkation_survival(df_clean, output_dir)
    plot_fare_boxplot(df_clean, output_dir)
    plot_correlation_heatmap(df_clean, output_dir)

    print("\nAll plots saved in the 'task2_outputs' folder.")


if __name__ == "__main__":
    main()
