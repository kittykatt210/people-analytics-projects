import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("workforce_kpis.csv")

for metric, title, ylab, fname in [
    ("headcount", "Monthly headcount trend", "Headcount", "headcount_trend.png"),
    ("engagement_index", "Employee engagement index", "Index", "engagement_index.png"),
    ("voluntary_attrition_rate", "Voluntary attrition rate", "Percent", "attrition_rate.png"),
]:
    plt.figure(figsize=(8, 4.6))
    plt.plot(df["month"], df[metric], marker="o")
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()
