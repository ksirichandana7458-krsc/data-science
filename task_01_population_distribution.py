import pandas as pd
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%201/API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv"


def main():
    
    df = pd.read_csv(DATA_URL, skiprows=4)

   
    year_columns = [col for col in df.columns if str(col).isdigit()]
    if not year_columns:
        raise ValueError("No year columns found in the dataset.")

    latest_year = max(year_columns, key=int)

    
    population = pd.to_numeric(df[latest_year], errors='coerce').dropna()

    
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#1b153f")
    ax.set_facecolor("#1b153f")

    ax.hist(population, bins=20, color="#5b7cff", edgecolor="#ffffff", alpha=0.9)
    ax.set_title("Distribution of Country Population", fontsize=20, color="white", pad=14)
    ax.set_xlabel(f"Population ({latest_year})", fontsize=13, color="white")
    ax.set_ylabel("Number of Countries", fontsize=13, color="white")
    ax.tick_params(axis="both", colors="white", labelsize=11)

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
