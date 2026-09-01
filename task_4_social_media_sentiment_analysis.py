"""Task 4: Sentiment analysis on social media data.

This script loads the Twitter sentiment dataset from the Prodigy-InfoTech
sample repository, cleans the text, analyzes sentiment patterns by entity,
and saves charts describing public opinion toward different topics/brands.
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_URL = (
    "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/"
    "main/Task%204/twitter_training.csv"
)


def load_data():
    df = pd.read_csv(DATASET_URL, header=None, names=["tweet_id", "entity", "sentiment", "text"], encoding="latin1")
    return df


def clean_data(df):
    
    df_clean = df.copy()

   
    df_clean = df_clean.dropna(subset=["text", "sentiment", "entity"]).copy()

  
    df_clean["sentiment"] = df_clean["sentiment"].str.strip().str.title()
    df_clean["entity"] = df_clean["entity"].astype(str).str.strip()
    df_clean["text"] = df_clean["text"].astype(str).str.strip()

  
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[@#]", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df_clean["clean_text"] = df_clean["text"].apply(clean_text)
    df_clean["text_length"] = df_clean["clean_text"].str.len()
    df_clean["word_count"] = df_clean["clean_text"].str.split().str.len()

    return df_clean


def print_overview(df):
    
    print("\n=== Dataset Overview ===")
    print(df.head().to_string(index=False))
    print("\nDataset shape:", df.shape)
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print("\nMissing values:")
    print(df.isnull().sum())


def show_top_entities(df, output_dir):
   
    entity_count = df["entity"].value_counts().head(10).reset_index()
    entity_count.columns = ["entity", "tweet_count"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=entity_count, x="tweet_count", y="entity", palette="viridis")
    plt.title("Top 10 Most Discussed Entities")
    plt.xlabel("Number of Tweets")
    plt.ylabel("Entity")
    plt.tight_layout()
    plt.savefig(output_dir / "top_entities.png")
    plt.close()

    print("\nTop 10 entities by tweet volume:")
    print(entity_count.to_string(index=False))


def plot_sentiment_distribution(df, output_dir):
    
    sentiment_order = ["Positive", "Neutral", "Negative", "Irrelevant"]
    sentiment_counts = df["sentiment"].value_counts().reindex(sentiment_order, fill_value=0)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_distribution.png")
    plt.close()

    print("\nSentiment counts:")
    print(sentiment_counts)


def plot_entity_sentiment_breakdown(df, output_dir):
    
    top_entities = df["entity"].value_counts().head(8).index.tolist()
    entity_sentiment = (
        df[df["entity"].isin(top_entities)]
        .groupby(["entity", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=entity_sentiment, x="entity", y="count", hue="sentiment", palette="deep")
    plt.title("Sentiment Mix by Top Entities")
    plt.xlabel("Entity")
    plt.ylabel("Tweet Count")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(output_dir / "entity_sentiment_breakdown.png")
    plt.close()


def analyze_sentiment_trends(df):
    
    entity_summary = (
        df.groupby("entity")
        .agg(
            total_tweets=("text", "count"),
            positive=("sentiment", lambda s: (s == "Positive").sum()),
            neutral=("sentiment", lambda s: (s == "Neutral").sum()),
            negative=("sentiment", lambda s: (s == "Negative").sum()),
            irrelevant=("sentiment", lambda s: (s == "Irrelevant").sum()),
        )
        .reset_index()
    )

    entity_summary["positive_share"] = entity_summary["positive"] / entity_summary["total_tweets"]
    entity_summary["negative_share"] = entity_summary["negative"] / entity_summary["total_tweets"]
    entity_summary = entity_summary.sort_values(["total_tweets", "positive_share"], ascending=[False, False])

    print("\n=== Entity Sentiment Summary ===")
    print(entity_summary.head(10).to_string(index=False))

    top_positive = entity_summary.sort_values("positive_share", ascending=False).head(5)
    top_negative = entity_summary.sort_values("negative_share", ascending=False).head(5)

    print("\nTop entities by positive share:")
    print(top_positive[["entity", "positive_share", "total_tweets"]].to_string(index=False))
    print("\nTop entities by negative share:")
    print(top_negative[["entity", "negative_share", "total_tweets"]].to_string(index=False))


def plot_text_length_distribution(df, output_dir):
  
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="sentiment", y="text_length", palette="pastel")
    plt.title("Tweet Length by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Text Length")
    plt.tight_layout()
    plt.savefig(output_dir / "tweet_length_by_sentiment.png")
    plt.close()


def main():
    output_dir = Path("task4_outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_data()
    df_clean = clean_data(df)

    print_overview(df_clean)
    plot_sentiment_distribution(df_clean, output_dir)
    show_top_entities(df_clean, output_dir)
    plot_entity_sentiment_breakdown(df_clean, output_dir)
    plot_text_length_distribution(df_clean, output_dir)
    analyze_sentiment_trends(df_clean)

    print("\nAll outputs were saved in the 'task4_outputs' folder.")


if __name__ == "__main__":
    main()
