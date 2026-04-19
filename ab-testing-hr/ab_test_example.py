import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

df = pd.read_csv("ab_test_sample.csv")
control = df.loc[df["group"] == "Control", "score"]
intervention = df.loc[df["group"] == "Intervention", "score"]

t_stat, p_val = ttest_ind(intervention, control, equal_var=False)
diff = intervention.mean() - control.mean()
se = np.sqrt(control.var(ddof=1) / len(control) + intervention.var(ddof=1) / len(intervention))
ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se

print(f"Mean difference: {diff:.2f}")
print(f"95% CI: ({ci_low:.2f}, {ci_high:.2f})")
print(f"p-value: {p_val:.4f}")
