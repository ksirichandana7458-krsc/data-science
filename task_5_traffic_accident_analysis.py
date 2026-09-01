"""Task 5: Traffic accident analysis on US accident data.

This script analyzes the US Accidents Kaggle dataset to identify patterns related
to road conditions, weather, and time of day. It also highlights accident hotspots
and the major contributing factors behind crashes.

Dataset source:
    https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_NAME = "sobhanmoosavi/us-accidents"
EXPECTED_FILE = "US_Accidents_March23.csv"
LOCAL_CANDIDATES = [
    Path("data") / EXPECTED_FILE,
    Path("US_Accidents_March23.csv"),
    Path("us_accidents.csv"),
    Path("data") / "us_accidents.csv",
]


def find_dataset_file() -> Path | None:
    """Locate the dataset in the workspace or download it if available via Kaggle API."""
    for candidate in LOCAL_CANDIDATES:
        if candidate.exists():
            return candidate

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    kaggle_cmd = ["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", str(data_dir), "--unzip"]
    try:
        subprocess.run(kaggle_cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    for candidate in [data_dir / EXPECTED_FILE, data_dir / "us_accidents.csv"]:
        if candidate.exists():
            return candidate

    zip_path = next(data_dir.glob("*.zip"), None)
    if zip_path:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        for candidate in [data_dir / EXPECTED_FILE, data_dir / "us_accidents.csv"]:
            if candidate.exists():
                return candidate

    return None


def load_data() -> pd.DataFrame:
    """Load the accidents dataset and validate required columns."""
    dataset_path = find_dataset_file()
    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset not found. Please download the Kaggle US Accidents data or place "
            "the CSV file in the project folder."
        )

    df = pd.read_csv(dataset_path)
    required = {"Start_Time", "City", "State", "Weather_Condition", "Road_Condition", "Severity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset for analysis."""
    data = df.copy()

    # Standardize date/time and text columns.
    data["Start_Time"] = pd.to_datetime(data["Start_Time"], errors="coerce")
    data["City"] = data["City"].fillna("Unknown").astype(str).str.strip()
    data["State"] = data["State"].fillna("Unknown").astype(str).str.strip()
    data["Weather_Condition"] = data["Weather_Condition"].fillna("Unknown").astype(str).str.strip()
    data["Road_Condition"] = data["Road_Condition"].fillna("Unknown").astype(str).str.strip()
    data["Severity"] = pd.to_numeric(data["Severity"], errors="coerce").fillna(0)

    data["Hour"] = data["Start_Time"].dt.hour
    data["Day_of_Week"] = data["Start_Time"].dt.day_name()
    data["Month"] = data["Start_Time"].dt.month_name()

    # Keep only relevant rows for analysis.
    data = data.dropna(subset=["Start_Time", "City", "State"]).copy()
    return data


def print_overview(df: pd.DataFrame) -> None:
    """Print summary statistics for the dataset."""
    print("\n=== Dataset Overview ===")
    print(df.head(5).to_string(index=False))
    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nTop cities by accident count:")
    print(df["City"].value_counts().head(10))
    print("\nSeverity distribution:")
    print(df["Severity"].value_counts().sort_index())


def plot_accident_hotspots(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the most accident-prone cities and states."""
    city_counts = df["City"].value_counts().head(10).reset_index()
    city_counts.columns = ["City", "Accidents"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=city_counts, x="Accidents", y="City", palette="viridis")
    plt.title("Top 10 Accident Hotspots by City")
    plt.xlabel("Number of Accidents")
    plt.ylabel("City")
    plt.tight_layout()
    plt.savefig(output_dir / "accident_hotspots_by_city.png", dpi=200)
    plt.close()

    state_counts = df["State"].value_counts().head(10).reset_index()
    state_counts.columns = ["State", "Accidents"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=state_counts, x="Accidents", y="State", palette="magma")
    plt.title("Top 10 States by Accident Count")
    plt.xlabel("Number of Accidents")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig(output_dir / "accident_hotspots_by_state.png", dpi=200)
    plt.close()

    print("\nTop 10 accident hotspots by city:")
    print(city_counts.to_string(index=False))


def plot_weather_patterns(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot accident distribution by weather condition."""
    weather_counts = df["Weather_Condition"].value_counts().head(10).reset_index()
    weather_counts.columns = ["Weather_Condition", "Accidents"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=weather_counts, x="Accidents", y="Weather_Condition", palette="coolwarm")
    plt.title("Accidents by Weather Condition")
    plt.xlabel("Number of Accidents")
    plt.ylabel("Weather")
    plt.tight_layout()
    plt.savefig(output_dir / "accidents_by_weather.png", dpi=200)
    plt.close()

    print("\nWeather conditions with highest accident counts:")
    print(weather_counts.to_string(index=False))


def plot_road_conditions(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot accident counts by road condition."""
    road_counts = df["Road_Condition"].value_counts().head(10).reset_index()
    road_counts.columns = ["Road_Condition", "Accidents"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=road_counts, x="Accidents", y="Road_Condition", palette="Set2")
    plt.title("Accidents by Road Condition")
    plt.xlabel("Number of Accidents")
    plt.ylabel("Road Condition")
    plt.tight_layout()
    plt.savefig(output_dir / "accidents_by_road_condition.png", dpi=200)
    plt.close()

    print("\nRoad conditions with highest accident counts:")
    print(road_counts.to_string(index=False))


def plot_time_of_day_patterns(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot accident frequency over the day."""
    hourly = df.groupby("Hour").size().reset_index(name="Accidents")

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly, x="Hour", y="Accidents", marker="o", color="royalblue")
    plt.title("Accident Trend by Time of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Accidents")
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(output_dir / "accidents_by_time_of_day.png", dpi=200)
    plt.close()

    print("\nAccident count by hour of day:")
    print(hourly.head(24).to_string(index=False))


def plot_severity_by_weather(df: pd.DataFrame, output_dir: Path) -> None:
    """Visualize average accident severity for different weather conditions."""
    severity_weather = (
        df.groupby("Weather_Condition", as_index=False)["Severity"]
        .mean()
        .sort_values("Severity", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=severity_weather, x="Severity", y="Weather_Condition", palette="rocket")
    plt.title("Average Accident Severity by Weather Condition")
    plt.xlabel("Average Severity")
    plt.ylabel("Weather Condition")
    plt.tight_layout()
    plt.savefig(output_dir / "avg_severity_by_weather.png", dpi=200)
    plt.close()

    print("\nAverage severity by weather condition:")
    print(severity_weather.to_string(index=False))


def summarize_key_findings(df: pd.DataFrame) -> None:
    """Print the key analytical conclusions from the cleaned dataset."""
    city_hotspot = df["City"].value_counts().idxmax()
    city_count = df["City"].value_counts().max()
    peak_hour = df["Hour"].value_counts().idxmax()
    most_common_weather = df["Weather_Condition"].value_counts().idxmax()
    most_common_road_condition = df["Road_Condition"].value_counts().idxmax()

    print("\n=== Key Findings ===")
    print(f"The most accident-prone city is {city_hotspot} with {city_count} accidents.")
    print(f"The peak hour for crashes is {peak_hour}:00.")
    print(f"The most frequent weather condition is {most_common_weather}.")
    print(f"The most common road condition is {most_common_road_condition}.")

    # Day-of-week risk pattern.
    weekday_counts = df["Day_of_Week"].value_counts().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ], fill_value=0)
    print("\nWeekday accident distribution:")
    print(weekday_counts)


def main() -> None:
    output_dir = Path("task5_outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_data()
    df_clean = clean_data(df)

    print_overview(df_clean)
    plot_accident_hotspots(df_clean, output_dir)
    plot_weather_patterns(df_clean, output_dir)
    plot_road_conditions(df_clean, output_dir)
    plot_time_of_day_patterns(df_clean, output_dir)
    plot_severity_by_weather(df_clean, output_dir)
    summarize_key_findings(df_clean)

    print(f"\nAll charts saved to the '{output_dir}' folder.")
    print("Use the generated plots to identify accident hotspots and contributory factors.")


if __name__ == "__main__":
    main()
