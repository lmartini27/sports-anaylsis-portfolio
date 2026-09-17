import pandas as pd

df = pd.read_csv("efficiency_ratings.csv")

print("Most extreme LOW net_rating rows:")
print(df.sort_values("net_rating").head(10)[["season", "team", "offensive_rating", "defensive_rating", "net_rating"]])

print("\nMost extreme HIGH net_rating rows:")
print(df.sort_values("net_rating", ascending=False).head(10)[["season", "team", "offensive_rating", "defensive_rating", "net_rating"]])

print("\nHow many rows are outside a plausible -40 to 40 range?")
print(((df["net_rating"] < -40) | (df["net_rating"] > 40)).sum(), "out of", len(df))
