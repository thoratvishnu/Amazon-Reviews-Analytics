"""Loading and cleaning for the Amazon consumer reviews dataset."""
import re
import pandas as pd

DATA_PATH = "data/reviews.csv"


def _product_type(text: str) -> str:
    t = text.lower()
    rules = [
        ("Fire TV", ["fire tv", "firetv"]),
        ("Echo / Alexa", ["echo", "alexa", "dot", "tap"]),
        ("Kindle", ["kindle", "paperwhite", "e-reader", "ebook"]),
        ("Fire Tablet", ["fire tablet", "fire hd", "fire kids", "tablet"]),
        ("Accessories", ["case", "cover", "charger", "adapter", "cable", "stand"]),
    ]
    for label, keys in rules:
        if any(k in t for k in keys):
            return label
    return "Other"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # drop columns that are empty / useless for analysis
    df = df.drop(columns=["reviews.userCity", "reviews.userProvince",
                          "reviews.id", "reviews.didPurchase", "keys"],
                 errors="ignore")
    df = df.rename(columns={
        "reviews.date": "date", "reviews.rating": "rating",
        "reviews.text": "text", "reviews.title": "title",
        "reviews.numHelpful": "helpful", "reviews.doRecommend": "recommend",
        "reviews.username": "username",
    })

    # --- product name: fill from other rows of the same product id ---
    def clean_name(n):
        if pd.isna(n):
            return None
        return re.split(r",,,|\n", str(n))[0].strip()

    df["name"] = df["name"].map(clean_name)
    df["name"] = df.groupby("id")["name"].transform(
        lambda s: s.fillna(s.dropna().iloc[0]) if s.notna().any() else s)
    df["product"] = df["name"].fillna("Unnamed (" + df["asins"].fillna("n/a") + ")")
    df["product_short"] = df["product"].str.slice(0, 45)

    # --- types / missing values ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["rating", "text"]).copy()
    df["rating"] = df["rating"].astype(int)
    df["helpful"] = df["helpful"].fillna(0).astype(int)
    df["recommend"] = df["recommend"].map({True: "Yes", False: "No", "True": "Yes", "False": "No"})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # --- features ---
    df["sentiment_label"] = pd.cut(df["rating"], [0, 2, 3, 5],
                                   labels=["Negative", "Neutral", "Positive"])
    df["review_length"] = df["text"].str.split().str.len()
    df["product_type"] = (df["product"].fillna("") + " " + df["categories"]).map(_product_type)
    df["brand"] = df["brand"].replace({"Amazon Fire Tv": "Amazon Fire TV"})
    return df.reset_index(drop=True)


def add_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Text sentiment score in [-1, 1] using VADER."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df["text_sentiment"] = df["text"].map(lambda t: sia.polarity_scores(str(t))["compound"])
    return df
