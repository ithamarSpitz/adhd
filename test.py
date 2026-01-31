import joblib
import pandas as pd

# Load the trained model pipeline
model = joblib.load('adhd_model.pkl')

# Define a new game session’s features (example values)
new_session = {"screws_spawned": 56, "screws_collected": 10, "forest_fish_collisions": 3, "forest_out_of_bounds": 7, "forest_5_screws_collected": 1, "forest_approaches": 59, "target_rt_median": 5.5, "target_rt_cv": 0.43, "forest_keys": 244, "forest_key_gap_mean": 1.67, "forest_key_gap_cv": 6.17, "river_fish_collisions": 3, "river_out_of_bounds": 7, "river_win": 1, "river_approaches": 54, "river_keys": 245, "river_key_gap_mean": 0.84, "river_key_gap_cv": 2.64, "dwarf_enters": 7, "dwarf_wm_rt_median": 1.73, "dwarf_wm_rt_cv": 3.48, "dwarf_correct": 12, "dwarf_incorrect": 8}

# Convert to DataFrame
df_new = pd.DataFrame([new_session])

# Predict probability of adhd
prob_adhd = model.predict_proba(df_new)[0][1]
print(f"Predicted probability of adhd: {prob_adhd:.2%}")