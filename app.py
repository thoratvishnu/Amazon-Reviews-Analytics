import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from data_utils import add_vader, load_data

st.set_page_config(page_title="Amazon Reviews Analytics", page_icon="📊", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1300px;}
.hero {background: linear-gradient(120deg,#232F3E,#FF9900); padding: 1.4rem 1.8rem;
       border-radius: 16px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem;} .hero p {margin: .3rem 0 0; opacity: .9;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.10); border-radius: 14px;
       padding: 14px 18px; border-left: 5px solid #FF9900;}
</style>
""", unsafe_allow_html=True)

COLORS = {"Positive": "#2ca02c", "Neutral": "#f2b705", "Negative": "#d62728"}
px.defaults.template = "plotly_white"


@st.cache_data(show_spinner="Loading data…")
def get_data():
    return add_vader(load_data())


@st.cache_data(show_spinner=False)
def top_words(texts: tuple, n: int = 15) -> pd.DataFrame:
    vec = CountVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(texts)
    counts = pd.Series(X.sum(axis=0).A1, index=vec.get_feature_names_out())
    return counts.nlargest(n).rename_axis("term").reset_index(name="count")


df = get_data()

# ---------- sidebar filters ----------
st.sidebar.header("🔎 Filters")
types = st.sidebar.multiselect("Product type", sorted(df.product_type.unique()),
                               default=sorted(df.product_type.unique()))
ratings = st.sidebar.slider("Rating range", 1, 5, (1, 5))
years = sorted(df.year.dropna().astype(int).unique())
yr = st.sidebar.select_slider("Review year", options=years, value=(years[0], years[-1]))
f = df[df.product_type.isin(types) & df.rating.between(*ratings)
       & (df.year.isna() | df.year.between(*yr))]
if f.empty:
    st.warning("No reviews match these filters.")
    st.stop()

# ---------- header + KPIs ----------
st.markdown("""<div class="hero"><h1>📊 Amazon Product Reviews Analytics</h1>
<p>Ratings, sentiment and customer voice across Kindle, Fire, Echo and Fire TV products</p></div>""",
            unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reviews", f"{len(f):,}")
k2.metric("Avg rating", f"{f.rating.mean():.2f} ⭐")
k3.metric("% Positive", f"{(f.sentiment_label == 'Positive').mean():.0%}")
k4.metric("Would recommend", f"{(f.recommend == 'Yes').sum() / max(f.recommend.notna().sum(), 1):.0%}")
k5.metric("Products", f"{f['product'].nunique()}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Products", "Sentiment", "Text insights", "Data explorer"])

# ---------- Overview ----------
with tab1:
    c1, c2 = st.columns(2)
    dist = f.rating.value_counts().sort_index().reset_index()
    dist.columns = ["rating", "reviews"]
    c1.plotly_chart(px.bar(dist, x="rating", y="reviews", text="reviews",
                           title="Rating distribution",
                           color_discrete_sequence=["#FF9900"]), use_container_width=True)
    c2.plotly_chart(px.pie(f, names="sentiment_label", hole=0.55, title="Sentiment split",
                           color="sentiment_label", color_discrete_map=COLORS),
                    use_container_width=True)
    trend = f.dropna(subset=["month"]).groupby("month").agg(
        reviews=("rating", "size"), avg_rating=("rating", "mean")).reset_index()
    st.plotly_chart(px.area(trend, x="month", y="reviews", title="Reviews over time",
                            color_discrete_sequence=["#232F3E"]), use_container_width=True)
    st.plotly_chart(px.line(trend, x="month", y="avg_rating", markers=True,
                            title="Average rating over time", range_y=[1, 5],
                            color_discrete_sequence=["#FF9900"]), use_container_width=True)

# ---------- Products ----------
with tab2:
    prod = f.groupby("product_short").agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                                          pct_negative=("sentiment_label", lambda s: (s == "Negative").mean())
                                          ).reset_index()
    min_rev = st.slider("Minimum reviews per product", 1, 500, 50)
    prod = prod[prod.reviews >= min_rev]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(prod.nlargest(10, "reviews").sort_values("reviews"), x="reviews",
                           y="product_short", orientation="h", title="Most reviewed products",
                           color_discrete_sequence=["#232F3E"]), use_container_width=True)
    c2.plotly_chart(px.bar(prod.nlargest(10, "avg_rating").sort_values("avg_rating"), x="avg_rating",
                           y="product_short", orientation="h", title="Highest rated products",
                           range_x=[1, 5], color_discrete_sequence=["#FF9900"]),
                    use_container_width=True)
    st.plotly_chart(px.scatter(prod, x="reviews", y="avg_rating", size="reviews", hover_name="product_short",
                               color="pct_negative", color_continuous_scale="RdYlGn_r",
                               title="Popularity vs satisfaction (colour = % negative)"),
                    use_container_width=True)
    ptype = f.groupby("product_type").rating.agg(["mean", "size"]).reset_index()
    st.plotly_chart(px.bar(ptype, x="product_type", y="mean", text_auto=".2f", range_y=[1, 5],
                           title="Average rating by product type",
                           color_discrete_sequence=["#FF9900"]), use_container_width=True)

# ---------- Sentiment ----------
with tab3:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(f, x="text_sentiment", nbins=40, color="sentiment_label",
                                 color_discrete_map=COLORS, title="VADER text sentiment score"),
                    use_container_width=True)
    c2.plotly_chart(px.box(f, x="rating", y="text_sentiment", title="Does review text match the star rating?",
                           color_discrete_sequence=["#232F3E"]), use_container_width=True)
    mismatch = f[(f.rating <= 2) & (f.text_sentiment > 0.5)]
    st.subheader(f"⚠️ Suspicious reviews: low rating but very positive text ({len(mismatch)})")
    st.dataframe(mismatch[["product_short", "rating", "title", "text", "text_sentiment"]].head(50),
                 use_container_width=True, hide_index=True)

# ---------- Text insights ----------
with tab4:
    pos, neg = f[f.rating >= 4].text, f[f.rating <= 2].text
    c1, c2 = st.columns(2)
    if len(pos) > 10:
        c1.plotly_chart(px.bar(top_words(tuple(pos)).sort_values("count"), x="count", y="term",
                               orientation="h", title="Top terms in 4–5★ reviews",
                               color_discrete_sequence=["#2ca02c"]), use_container_width=True)
    if len(neg) > 10:
        c2.plotly_chart(px.bar(top_words(tuple(neg)).sort_values("count"), x="count", y="term",
                               orientation="h", title="Top terms in 1–2★ reviews",
                               color_discrete_sequence=["#d62728"]), use_container_width=True)
    if len(pos) <= 10 or len(neg) <= 10:
        st.info("Widen the rating filter to compare positive and negative terms.")
    st.plotly_chart(px.box(f, x="rating", y="review_length", log_y=True,
                           title="Review length (words) by rating",
                           color_discrete_sequence=["#FF9900"]), use_container_width=True)
    st.subheader("🏆 Most helpful reviews")
    st.dataframe(f.nlargest(15, "helpful")[["product_short", "rating", "helpful", "title", "text"]],
                 use_container_width=True, hide_index=True)

# ---------- Data explorer ----------
with tab5:
    q = st.text_input("Search review text", placeholder="e.g. battery, screen, kids…")
    view = f[f.text.str.contains(q, case=False, na=False)] if q else f
    st.caption(f"{len(view):,} reviews")
    cols = ["date", "product_short", "rating", "recommend", "title", "text", "helpful"]
    st.dataframe(view[cols].head(1000), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data (CSV)", view.to_csv(index=False).encode(),
                       "filtered_reviews.csv", "text/csv")
