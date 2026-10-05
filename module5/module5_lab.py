# Module 5 Lab 1: Chart It, Then Test It
# CS 82A Intro to Data Science | Azra (Ramisha Tasfia)
# Repo: https://github.com/ramishatasfia5156-commits/cs82a-coursework/tree/main/module5
#
# VS Code version. Run cell by cell (# %% markers) or the whole file.
# ONE-TIME SETUP: run this line in the VS Code terminal first:
#     python3 -m pip install seaborn scipy

# %% Setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

sns.set_theme(style="whitegrid")

# %% Step 1: Group means for chick feeds
chicks = pd.read_csv("chickwts.csv")
feed_means = chicks.groupby("feed")["weight"].mean().sort_values()
print(feed_means.round(1))

# %% Step 2: Sorted bar chart (order = lowest to highest mean)
sns.barplot(data=chicks, x="feed", y="weight", order=feed_means.index,
            errorbar=None, color="steelblue")
plt.title("Sunflower-fed chicks weigh the most; horsebean-fed the least")
plt.xlabel("Feed type")
plt.ylabel("Mean weight (grams)")
plt.ylim(bottom=0)
plt.show()

# %% Step 3: Histogram of all weights
sns.histplot(data=chicks, x="weight", bins=15, color="steelblue")
plt.title("Chick weights range 108-423 g, centered near 260 g")
plt.xlabel("Weight (grams)")
plt.ylabel("Number of chicks")
plt.show()

# %% Step 4: Line chart of museum visitors
museum = pd.read_csv("museum_visitors.csv", parse_dates=["Date"])
museum_long = museum.melt(id_vars="Date", var_name="Museum", value_name="Visitors")

plt.figure(figsize=(11, 5))
sns.lineplot(data=museum_long, x="Date", y="Visitors", hue="Museum")
plt.title("Avila Adobe draws the most visitors, with a summer peak each year (2014-2018)")
plt.xlabel("Month")
plt.ylabel("Visitors per month")
plt.ylim(bottom=0)
plt.legend(title="Museum", fontsize=8)
plt.show()
# Note: Firehouse Museum's Sept 2014 spike (61,192) is a one-month outlier.

# %% Step 5a: Load Module 4 sales data and build revenue
sales = pd.read_csv("sales_clean.csv")
print("Rows with negative qty (returns or entry errors):", (sales["qty"] < 0).sum())
sales = sales[sales["qty"] > 0].copy()          # drop invalid negative quantities
sales["revenue"] = sales["price"] * sales["qty"]
print(sales[["price", "qty", "revenue"]].describe().round(2))

# %% Step 5b: Scatter qty vs. revenue
sns.scatterplot(data=sales, x="qty", y="revenue", alpha=0.6)
plt.title("Bigger orders earn somewhat more, but price drives revenue")
plt.xlabel("Quantity per order (units)")
plt.ylabel("Revenue per order (USD)")
plt.show()

# %% Step 5c: Correlation heatmap
corr = sales[["price", "qty", "revenue"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Revenue tracks price far more than quantity")
plt.show()
print(corr.round(2))

# %% Step 7: t-test, sunflower vs. horsebean
sunflower = chicks[chicks["feed"] == "sunflower"]["weight"]
horsebean = chicks[chicks["feed"] == "horsebean"]["weight"]
print("n sunflower:", len(sunflower), "| n horsebean:", len(horsebean))

result = ttest_ind(sunflower, horsebean)
diff = sunflower.mean() - horsebean.mean()
print(f"Mean sunflower: {sunflower.mean():.1f} g")
print(f"Mean horsebean: {horsebean.mean():.1f} g")
print(f"Difference (effect size): {diff:.1f} g")
print(f"t = {result.statistic:.2f}, p = {result.pvalue:.2e}")
