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

def simulate_month():
    print("Starting 30-Day Simulation...")
    
    recipes_df, _, _ = flc.load_data()
    
    # Simulation Start Date (Tomorrow)
    start_date = datetime.now() + timedelta(days=1)
    
    slots = ["Breakfast", "Lunch", "Dinner"]
    
    # Open Report File once
    with open(REPORT_FILE, "w") as report:
        report.write("=== COMPREHENSIVE 30-DAY MEAL PLAN & AUDIT ===\n")
        report.write("Key:\n")
        report.write("  Similarity: Final Score (0.0 - 1.0). Higher is better.\n")
        report.write("  Match(Dist^2): Raw deviation from Target. Lower is better (0=Perfect).\n")
        report.write("  Penalty: Repetition decay. Higher means recently eaten.\n\n")
        
        # Initial Params
        with open(PARAMS_FILE, 'r') as f:
            initial_params = json.load(f)
        report.write(f"Initial Spice Weight: {initial_params['weights']['spice']:.4f}\n\n")
        
        for day in range(30):  # Changed from 7 to 30
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
                    
                    # Re-calc Penalty component (pass current date)
                    penalty = flc.calculate_decay_penalty(row['recipe_id'], row.get('cuisine'), history_df, slot, user_params, current_date_base)
                    
                    # Derive Distance (note: with new formula, this is approximate)
                    # Old: total_dissim = D² + P, New: S = (1/(1+D²)) * exp(-γP)
                    # For reporting, we'll approximate D² from the base similarity
                    gamma = user_params.get('gamma', 0.8)
                    if penalty > 0:
                        s_base = score / np.exp(-gamma * penalty)
                    else:
                        s_base = score
                    dist_sq = (1.0 / s_base) - 1.0 if s_base > 0 else 0.0
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


    # Post-simulation analysis: Track most repeated recipe
    print("\n[*] Analyzing repetition patterns...")
    
    # Load final history
    final_history = pd.read_csv(HISTORY_FILE)
    final_history['date'] = pd.to_datetime(final_history['date'])
    
    # IMPORTANT: Only count recipes from THIS simulation (after start_date)
    simulation_history = final_history[final_history['date'] >= start_date]
    
    # Find most repeated recipe IN THIS SIMULATION
    from collections import Counter
    recipe_counts = Counter(simulation_history['recipe_id'])
    if recipe_counts:
        most_repeated_id = recipe_counts.most_common(1)[0][0]
        repeat_count = recipe_counts.most_common(1)[0][1]
        
        # Get recipe name
        recipe_name = recipes_df[recipes_df['recipe_id'] == most_repeated_id]['title'].iloc[0]
        
        print(f"Most repeated: {recipe_name} ({repeat_count} times)")
        
        # Track this recipe across all days
        with open(REPORT_FILE, "a") as report:
            report.write("\n\n" + "="*80 + "\n")
            report.write("CONTINUOUS TRACKING: Most Repeated Recipe\n")
            report.write("="*80 + "\n")
            report.write(f"Recipe: {recipe_name} (ID: {most_repeated_id})\n")
            report.write(f"Total selections: {repeat_count}\n\n")
            report.write("Day-by-Day Evolution:\n")
            report.write("Format: Day | Distance(D²) | Penalty(P) | Score(S) | Status\n")
            report.write("-" * 80 + "\n")
            
            # Reload params for consistency
            with open(PARAMS_FILE, 'r') as f:
                final_params = json.load(f)
            
            # For each day, calculate what the distance/penalty/score would have been
            for day in range(1, 31):
                current_date_sim = start_date + timedelta(days=day-1)
                
                # Get history up to this day
                history_up_to_day = final_history[final_history['date'] < current_date_sim]
                
                # Build profile and target (using Dinner as example slot)
                safe_recipes_sim = flc.filter_safe_recipes(recipes_df)
                profile_sim = flc.build_user_profile(safe_recipes_sim, history_up_to_day)
                target_sim = flc.run_fuzzy_systems(profile_sim, final_params, "Dinner", safe_recipes_sim)
                
                # Get the specific recipe
                recipe_row = recipes_df[recipes_df['recipe_id'] == most_repeated_id].iloc[0]
                
                # Calculate distance
                features_map = {
                    'calories': 'calories', 'protein': 'protein_g', 'carbs': 'carbohydrates_g',
                    'fat': 'fat_g', 'prep_time': 'prep_time_minutes', 'spice': 'spiciness_score'
                }
                dist_sq = 0
                for key, col in features_map.items():
                    t_val = target_sim[key]
                    actual = recipe_row[col]
                    w = final_params['weights'].get(key, 0.1)
                    std = recipes_df[col].std()
                    if std == 0: std = 1
                    diff = (actual - t_val) / std
                    dist_sq += w * (diff ** 2)
                
                # Calculate penalty
                penalty = flc.calculate_decay_penalty(
                    most_repeated_id, 
                    recipe_row.get('cuisine', ''), 
                    history_up_to_day, 
                    "Dinner", 
                    final_params,
                    current_date_sim
                )
                
                # Calculate score
                gamma = final_params.get('gamma', 0.8)
                s_base = 1.0 / (1.0 + dist_sq)
                score = s_base * np.exp(-gamma * penalty)
                
                # Check if selected on this day
                day_selections = final_history[
                    (final_history['date'].dt.date == current_date_sim.date()) &
                    (final_history['recipe_id'] == most_repeated_id)
                ]
                status = "SELECTED" if not day_selections.empty else "not selected"
                
                report.write(f"{day:3d} | {dist_sq:12.4f} | {penalty:10.4f} | {score:10.6f} | {status}\n")
            
            report.write("="*80 + "\n")
    
    print(f"\nSimulation Complete. Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    setup_simulation()
    simulate_month()  # Changed from simulate_week
