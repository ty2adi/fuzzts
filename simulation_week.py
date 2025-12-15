import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import fuzzy_poc as flc  # Import the FLC module

# File Paths
RECIPES_FILE = "recipes_rows_spice0_100_minimal.csv"
HISTORY_FILE = "mock_input_simulation.csv"
PARAMS_FILE = "user_params.json"
REPORT_FILE = "simulation_report.txt"

def setup_simulation():
    # 1. Reset History to a small seed (or empty) to start fresh for the week
    # For a realistic week, let's start with a few random items so profile isn't totally empty
    # or just copy the existing mock mock_input_table.csv as a baseline.
    try:
        base_df = pd.read_csv("mock_input_table.csv")
        base_df['date'] = pd.to_datetime(base_df['date'])
        
        # Truncate to just the "Past" so we can simulate "Next Week"
        # Let's keep 10 rows.
        base_df = base_df.head(10)
        base_df.to_csv(HISTORY_FILE, index=False)
        print(f"Initialized {HISTORY_FILE} with {len(base_df)} rows.")
    except:
        print("No baseline history found. Starting empty.")
        pd.DataFrame(columns=['user_id', 'date', 'meal_slot', 'recipe_id']).to_csv(HISTORY_FILE, index=False)

    # 2. Reset Params (Optional, but good to see learning from scratch)
    # We'll keep existing params to continue learning.
    pass

def simulate_week():
    print("Starting 7-Day Simulation...")
    
    recipes_df, _, _ = flc.load_data()
    
    # Simulation Start Date (Tomorrow)
    start_date = datetime.now() + timedelta(days=1)
    
    slots = ["Breakfast", "Lunch", "Dinner"]
    
    # Open Report File once
    with open(REPORT_FILE, "w") as report:
        report.write("=== COMPREHENSIVE WEEKLY MEAL PLAN & AUDIT ===\n")
        report.write("Key:\n")
        report.write("  Similarity: Final Score (0.0 - 1.0). Higher is better.\n")
        report.write("  Match(Dist^2): Raw deviation from Target. Lower is better (0=Perfect).\n")
        report.write("  Penalty: Repetition decay. Higher means recently eaten.\n\n")
        
        # Initial Params
        with open(PARAMS_FILE, 'r') as f:
            initial_params = json.load(f)
        report.write(f"Initial Spice Weight: {initial_params['weights']['spice']:.4f}\n\n")
        
        for day in range(7):
            current_date_base = start_date + timedelta(days=day)
            date_str = current_date_base.strftime('%Y-%m-%d')
            print(f"\n--- Day {day+1} ({date_str}) ---")
            report.write(f"--- DAY {day+1}: {date_str} ---\n")
            
            for slot in slots:
                # 1. Reload Dynamic Data
                history_df = pd.read_csv(HISTORY_FILE)
                history_df['date'] = pd.to_datetime(history_df['date'])
                
                with open(PARAMS_FILE, 'r') as f:
                    user_params = json.load(f)
                    
                # 2. Run FLC Pipeline
                safe_recipes = flc.filter_safe_recipes(recipes_df)
                profile = flc.build_user_profile(safe_recipes, history_df)
                target = flc.run_fuzzy_systems(profile, user_params, slot, safe_recipes)
                
                # Ranked Candidates
                ranked = flc.rank_recipes(safe_recipes, target, user_params, slot, history_df)
                
                if ranked.empty:
                    report.write(f"  [{slot}] No recipes found.\n")
                    continue
                    
                report.write(f"  [{slot}] Target: Cal={target['calories']:.0f}, Spice={target['spice']:.0f}\n")
                
                # Analyze Top 5 Candidates
                top_rec = ranked.iloc[0]
                
                for i, (idx, row) in enumerate(ranked.iterrows()):
                    title = row['title']
                    score = row['score']
                    
                    # Deconstruct Score: Score = 1 / (1 + Dissim)
                    total_dissim = (1.0 / score) - 1.0
                    
                    # Re-calc Penalty component
                    penalty = flc.calculate_decay_penalty(row['recipe_id'], row.get('cuisine'), history_df)
                    
                    # Derive Weights Dist
                    dist_sq = total_dissim - penalty
                    if dist_sq < 0: dist_sq = 0.0
                    
                    status = "[SELECTED]" if i == 0 else "[REJECTED]"
                    
                    # Explain rejection
                    reason = ""
                    if i > 0:
                        if penalty > 0.5:
                            reason = " [High Recency Penalty]"
                        elif dist_sq > 0.5:
                            reason = " [Poor Target Match]"
                        else:
                            reason = " [Better option exists]"
                            
                    report.write(f"    #{i+1} {status}: {title}\n")
                    report.write(f"       Sim: {score:.4f} | Match(Dist^2): {dist_sq:.4f} | Penalty: {penalty:.4f}{reason}\n")
                    
                report.write("\n")
                
                # 4. Feedback Loop
                new_row = {
                    'user_id': 'sim_user',
                    'date': current_date_base.replace(hour=8 if slot=='Breakfast' else 13 if slot=='Lunch' else 19),
                    'meal_slot': slot,
                    'recipe_id': top_rec['recipe_id']
                }
                new_history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
                new_history_df.to_csv(HISTORY_FILE, index=False)
                
                # 5. Training
                new_params = flc.train_weights(user_params, new_history_df, recipes_df, target)
                flc.save_params(new_params)

    print(f"\nSimulation Complete. Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    setup_simulation()
    simulate_week()
