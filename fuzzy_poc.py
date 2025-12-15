import pandas as pd
import numpy as np
import json
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime

# --- 1. Load Data ---
def load_data():
    recipes = pd.read_csv("recipes_rows_spice0_100_minimal.csv")
    try:
        input_table = pd.read_csv("mock_input_table.csv")
        # Ensure date parsing
        if 'date' in input_table.columns and input_table['date'].dtype == 'object':
            input_table['date'] = pd.to_datetime(input_table['date'])
    except Exception:
        input_table = pd.DataFrame() # Handle missing history gracefully
        
    try:
        with open("user_params.json", "r") as f:
            user_params = json.load(f)
    except FileNotFoundError:
        # Default params if missing
        user_params = {
            "weights": {
                "calories": 0.25, "protein": 0.2, "carbs": 0.2, "fat": 0.15, "prep_time": 0.1, "spice": 0.1
            },
            "deltas": { "spice_peak": 0.0, "prep_boundary": 0.0 }
        }
    return recipes, input_table, user_params

def save_params(params):
    with open("user_params.json", "w") as f:
        json.dump(params, f, indent=4)

# --- X. Safety Filter (Stage A) ---
def filter_safe_recipes(recipes_df):
    """
    Stage A: Safety Filter.
    Removes recipes incompatible with user constraints (Allergies, etc.).
    For POC, we assume all valid recipes in DB are safe, 
    but we define this explicit step for pipeline compliance.
    """
    # Example: Filter where 'is_active' is True if that column exists
    if 'is_active' in recipes_df.columns:
        return recipes_df[recipes_df['is_active'] == True].copy()
    return recipes_df.copy()

# --- 2. Profile Builder (Stage B) ---
def build_user_profile(recipes_df, input_table_df):
    """
    Stage B: Profile Builder with Cold-Start Shrinkage.
    Formula: pref = (n/(n+k))*pref_user + (k/(n+k))*pref_global
    k = 10
    """
    # Global Stats (for shrinkage fallback)
    global_medians = {
        'calories': recipes_df['calories'].median(),
        'protein': recipes_df['protein_g'].median(),
        'carbs': recipes_df['carbohydrates_g'].median(),
        'fat': recipes_df['fat_g'].median(),
        'prep_time': recipes_df['prep_time_minutes'].median(),
        'spice': recipes_df['spiciness_score'].median()
    }

    if input_table_df.empty:
        return global_medians

    # Get User History (Last 21)
    recent_history = input_table_df.sort_values('date', ascending=False).head(21)
    n = len(recent_history)
    k = 10.0 # Shrinkage factor
    
    # Merge to get nutrient data
    # Note: If history is just simulation, we need to merge with recipes_df to get values
    merged = recent_history.merge(recipes_df, on='recipe_id', how='left')
    
    user_medians = {
        'calories': merged['calories'].median(),
        'protein': merged['protein_g'].median(),
        'carbs': merged['carbohydrates_g'].median(),
        'fat': merged['fat_g'].median(),
        'prep_time': merged['prep_time_minutes'].median(),
        'spice': merged['spiciness_score'].median()
    }
    
    # Apply Shrinkage Blending
    final_profile = {}
    weight_user = n / (n + k)
    weight_global = k / (n + k)
    
    for key, g_val in global_medians.items():
        u_val = user_medians.get(key)
        if pd.isna(u_val): u_val = g_val # Handle missing data
            
        final_profile[key] = (weight_user * u_val) + (weight_global * g_val)
            
    return final_profile

# --- 3. Fuzzy Logic Engine (Full Mamdani) ---
def get_dynamic_universe(recipes_df, col_name, delta=0.0):
    """
    Constructs Fuzzy Sets (Low, Med, High) based on data percentiles.
    Applies 'delta' shift to the internal boundaries.
    """
    # 1. Stats
    min_v = recipes_df[col_name].min()
    max_v = recipes_df[col_name].max()
    p20 = recipes_df[col_name].quantile(0.20) + delta
    p50 = recipes_df[col_name].quantile(0.50) + delta
    p80 = recipes_df[col_name].quantile(0.80) + delta
    
    # Safety: Ensure order min <= p20 <= p50 <= p80 <= max
    # (Simple clamping to avoid fuzzy errors)
    p20 = max(min_v, min(p20, max_v))
    p50 = max(p20, min(p50, max_v))
    p80 = max(p50, min(p80, max_v))
    
    # 2. Variable
    # Universe range: slightly padded
    safe_max = max(max_v, 100) # Ensure at least reasonable range
    x_axis = np.linspace(min_v, safe_max, 100)
    
    # 3. Membership Functions
    # Low: Trap [min, min, p20, p50]
    low = fuzz.trapmf(x_axis, [min_v, min_v, p20, p50])
    # Med: Tri [p20, p50, p80]
    med = fuzz.trimf(x_axis, [p20, p50, p80])
    # High: Trap [p50, p80, max, max]
    high = fuzz.trapmf(x_axis, [p50, p80, safe_max, safe_max])
    
    return x_axis, low, med, high

def solve_fuzzy_variable(name, col_name, pref_val, meal_slot, recipes_df, deltas):
    """
     Solves Mamdani Inference for a single variable.
     Returns: Defuzzified Target Value (Centroid).
    """
    # 1. Get Delta (only for specific vars)
    delta = 0.0
    if name == 'spice': delta = deltas.get('spice_peak', 0.0)
    if name == 'prep_time': delta = deltas.get('prep_boundary', 0.0)
    
    x_axis, mf_low, mf_med, mf_high = get_dynamic_universe(recipes_df, col_name, delta)
    
    # 2. Fuzzify Inputs (Preference)
    # Calculate degree of membership for the USER's profile value
    # interp_membership works on the x_axis
    mu_pref_low = fuzz.interp_membership(x_axis, mf_low, pref_val)
    mu_pref_med = fuzz.interp_membership(x_axis, mf_med, pref_val)
    mu_pref_high = fuzz.interp_membership(x_axis, mf_high, pref_val)
    
    # 3. Evaluation of Rules
    # We aggregate firing strengths for Low, Med, High outputs.
    
    # A. Preference Rules (Copy Input -> Output)
    # "If Pref Low Then Target Low"
    fire_low = mu_pref_low
    fire_med = mu_pref_med
    fire_high = mu_pref_high
    
    # B. Slot Rules (The "Nudges")
    # Rule Base from Spec:
    # • Breakfast -> Cal High
    # • Lunch -> Cal Med
    # • Dinner -> Cal Low, Prep Low, Spice Mild
    
    # Logic: OR (Max) logic. If Slot Rule says "High", we add 1.0 strength to High?
    # Actually, Slot rules are usually constraints. "If Dinner, Target IS Low".
    # This implies usually we want to force it. Mamdani 'MAX' aggregation means
    # if Slot says "Target is Low" (Strength 1.0), result will be strongly Low.
    
    if name == 'calories':
        if meal_slot == 'Breakfast':
            fire_high = max(fire_high, 1.0) # Boost High
        elif meal_slot == 'Lunch':
            fire_med = max(fire_med, 1.0)
        elif meal_slot == 'Dinner':
            fire_low = max(fire_low, 1.0)
            
    if name == 'prep_time':
        if meal_slot == 'Dinner':
            fire_low = max(fire_low, 1.0)
            
    if name == 'spice':
        if meal_slot == 'Dinner':
            fire_low = max(fire_low, 1.0) # Mild
            
    # 4. Aggregation (Max) and Defuzzification (Centroid)
    # Cut the MFs at the firing levels
    # Low area
    out_low = np.fmin(fire_low, mf_low)
    out_med = np.fmin(fire_med, mf_med)
    out_high = np.fmin(fire_high, mf_high)
    
    # Aggregate
    aggregated = np.fmax(out_low, np.fmax(out_med, out_high))
    
    # Defuzzify
    # Handle zero area edge case (shouldn't happen with 1.0 boosts but possible)
    if np.sum(aggregated) == 0:
        return pref_val # Fallback to profile
        
    target_val = fuzz.defuzz(x_axis, aggregated, 'centroid')
    return target_val

def run_fuzzy_systems(profile, user_params, meal_slot, recipes_df):
    """
    Runs the logic for ALL 6 variables.
    """
    variables = {
        'calories': 'calories',
        'protein': 'protein_g',
        'carbs': 'carbohydrates_g',
        'fat': 'fat_g',
        'prep_time': 'prep_time_minutes',
        'spice': 'spiciness_score'
    }
    
    target_vector = {}
    
    for var_name, col_name in variables.items():
        pref = profile[var_name]
        val = solve_fuzzy_variable(
            var_name, 
            col_name, 
            pref, 
            meal_slot, 
            recipes_df, 
            user_params['deltas']
        )
        target_vector[var_name] = val
        
    return target_vector

# --- 4. Ranker ---
def calculate_decay_penalty(recipe_id, cuisine, input_table_df, meal_slot=None):
    """
    Stage E: Repetition Control using Decay Penalties.
    P_recipe = Ar * exp(-dr/3)
    P_cuisine = Ac * exp(-dc/2)
    
    Update: Streak Penalty. 
    If recipe eaten 2 consecutive times in same slot -> Ar *= 10.
    """
    if input_table_df.empty: return 0.0
    now = input_table_df['date'].max()
    
    # 1. Base P_recipe
    # Force string comparison for safety
    rid_str = str(recipe_id)
    input_table_df['recipe_id_str'] = input_table_df['recipe_id'].astype(str)
    
    recipe_matches = input_table_df[input_table_df['recipe_id_str'] == rid_str]
    p_recipe = 0.0
    
    if not recipe_matches.empty:
        last_date = recipe_matches['date'].max()
        days_diff = (now - last_date).total_seconds() / 86400.0
        
        # Base Amplitude
        # User Feedback: Distances are ~30-40, so we need large penalties to have effect.
        # A=5.0 ensures regular repeats are pushed down if close alternative exists.
        A_r = 5.0
        
        # Streak Check
        if meal_slot:
            # Get chronological history for this slot
            slot_history = input_table_df[input_table_df['meal_slot'] == meal_slot].sort_values('date', ascending=False)
            
            # Check last 2 unique orders? No, strictly last 2 entries.
            if len(slot_history) >= 2:
                last_1 = str(slot_history.iloc[0]['recipe_id'])
                last_2 = str(slot_history.iloc[1]['recipe_id'])
                
                # If streak (last 1 & 2 matches this recipe)
                if last_1 == rid_str and last_2 == rid_str:
                    # Streak! Increase penalty duration/magnitude.
                    # Math: To ban for 2 weeks (14 days), we need P > ~2.0 at day 14.
                    # 100 * exp(-14/3) = 100 * 0.009 = 0.9.
                    # So A=100 ensures ~2 week "timeout".
                    A_r = 100.0
                    
        p_recipe = A_r * np.exp(-days_diff / 3.0)
    
    # 2. P_cuisine
    p_cuisine = 0.0 
    
    return p_recipe + p_cuisine

def rank_recipes(recipes_df, target_vector, user_params, meal_slot, input_table_df):
    candidates = recipes_df[recipes_df['meal_type'].str.lower() == meal_slot.lower()].copy()
    if candidates.empty:
        print(f"No recipes found for slot {meal_slot}.")
        return candidates
        
    weights = user_params['weights']
    
    # Pre-calculate cuisine history dates for efficiency?
    history_with_cuisine = input_table_df.merge(recipes_df[['recipe_id', 'cuisine']], on='recipe_id', how='left')
    
    def get_cuisine_penalty(target_cuisine, history_df, now):
        if pd.isna(target_cuisine): return 0.0
        matches = history_df[history_df['cuisine'] == target_cuisine]
        if matches.empty: return 0.0
        last_date = matches['date'].max()
        days_diff = (now - last_date).total_seconds() / 86400.0
        return 0.35 * np.exp(-days_diff / 2.0)

    now = input_table_df['date'].max() if not input_table_df.empty else datetime.now()

    features_map = {
        'calories': 'calories', 'protein': 'protein_g', 'carbs': 'carbohydrates_g',
        'fat': 'fat_g', 'prep_time': 'prep_time_minutes', 'spice': 'spiciness_score'
    }
    
    scores = []
    for idx, row in candidates.iterrows():
        dist_sq = 0
        for key, col in features_map.items():
            t_val = target_vector[key]
            actual = row[col]
            w = weights.get(key, 0.1)
            
            # Std deviation for norm
            std = recipes_df[col].std()
            if std == 0: std = 1
            
            diff = (actual - t_val) / std
            dist_sq += w * (diff ** 2)
            
        # Penalties
        penalty = calculate_decay_penalty(row['recipe_id'], row.get('cuisine', ''), input_table_df, meal_slot)
        
        # P_cuisine add-on
        p_c = get_cuisine_penalty(row.get('cuisine'), history_with_cuisine, now)
        penalty += p_c
        
        # Similarity Score
        similarity = 1.0 / (1.0 + dist_sq + penalty)
        scores.append(similarity)
        
    candidates['score'] = scores
    return candidates.sort_values('score', ascending=False).head(5)

# --- 5. Training ---
def train_weights(user_params, history_df, recipes_df, target_vector):
    if history_df.empty: return user_params
    
    last = history_df.iloc[0]
    chosen = recipes_df[recipes_df['recipe_id'] == last['recipe_id']].iloc[0]
    print(f"Training on last choice: ID {last['recipe_id']} ({chosen['title']})")
    
    # Low Learning Rates to prevent overfitting
    LR_W = 0.005
    LR_D = 0.05  # Reduced from 0.5 to 0.05 as requested
    
    features_map = {
        'calories': 'calories', 'protein': 'protein_g', 'carbs': 'carbohydrates_g',
        'fat': 'fat_g', 'prep_time': 'prep_time_minutes', 'spice': 'spiciness_score'
    }
    
    for key, col in features_map.items():
        actual = chosen[col]
        pred = target_vector[key]
        err = actual - pred
        
        # Update Deltas (only spice/prep)
        if key == 'spice':
            if err > 10: user_params['deltas']['spice_peak'] += LR_D
            elif err < -10: user_params['deltas']['spice_peak'] -= LR_D
        if key == 'prep_time':
            if err > 10: user_params['deltas']['prep_boundary'] += LR_D
            elif err < -10: user_params['deltas']['prep_boundary'] -= LR_D
            
        # Update Weights
        # Error Ratio
        denom = pred if pred != 0 else 1
        ratio = abs(err) / abs(denom)
        
        # If significant deviation, maybe increase weight?
        if ratio > 0.25:
             if user_params['weights'][key] < 0.8:
                user_params['weights'][key] += LR_W
        elif ratio < 0.05:
             if user_params['weights'][key] > 0.05:
                user_params['weights'][key] -= LR_W / 2
                
    return user_params

# --- 6. Math Explanation Report ---
def generate_math_report(profile, target_vector, user_params, chosen_recipe, new_params):
    print("\n" + "="*40)
    print("      FLC MATHEMATICAL DERIVATION      ")
    print("="*40)
    
    print("\n1. PROFILE GENERATION")
    print("   P_x = Median((Order_{t-1}, ..., Order_{t-21})_x)")
    print(f"   Ex: P_spice = {profile['spice']:.1f}")

    print("\n2. FUZZIFICATION & INFERENCE (Example: Spice)")
    print("   Universe U = [0, 100]")
    peak = user_params['deltas']['spice_peak']
    print(f"   Membership Shifts: Delta_spice = {peak:.2f}")
    print("   Sets: Low(x), Med(x), High(x) defined by dynamic percentiles.")
    print("   Input: P_spice")
    print("   Rules applied:")
    print("     R1: If P_spice is Low/Med/High -> Target is Low/Med/High")
    print("     R2: If Meal is Dinner -> Target is Mild (Force Low)")
    print("   Aggregation: mu_agg(x) = max(mu_R1, mu_R2)")
    print(f"   Defuzzification (Centroid): T_spice = Integral(x * mu_agg) / Integral(mu_agg)")
    print(f"   Result: T_spice = {target_vector['spice']:.1f}")

    print("\n3. SCORING & RANKING")
    print("   For recipe r, feature i:")
    print("   Normalized Diff:  z_i = (r_i - T_i) / sigma_i")
    print("   Weighted Dist:    D^2(r) = Sum( w_i * z_i^2 )")
    print("   Repetition Penalty: P(r) = 1.0 * exp(-days_since_eaten / 3.0)")
    print("   Final Similarity: S(r) = 1 / (1 + D^2(r) + P(r))")
    
    print("\n4. TRAINING (WEIGHT UPDATE)")
    print(f"   Actual Choice: '{chosen_recipe['title']}' (Spice={chosen_recipe['spiciness_score']})")
    print(f"   Error E_spice = Actual - Target = {chosen_recipe['spiciness_score']:.1f} - {target_vector['spice']:.1f} = {chosen_recipe['spiciness_score'] - target_vector['spice']:.1f}")
    print("   Update Logic:")
    print("   Delta_{t+1} = Delta_t + eta_d * (Error)")
    print("   w_{t+1} = w_t + eta_w * |Error|")
    print(f"   Updated Spice Weight: {user_params['weights']['spice']:.4f} -> {new_params['weights']['spice']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    print("=== FUZZY POC START ===")
    recipes, history, params = load_data()
    
    # Stage A: Safety
    safe_recipes = filter_safe_recipes(recipes)
    
    # 1. Profile
    profile = build_user_profile(safe_recipes, history)
    print("Profile:", {k: round(v,1) for k,v in profile.items()})
    
    # 2. Engine
    slot = "Dinner"
    print(f"Generating Target for {slot}...")
    target = run_fuzzy_systems(profile, params, slot, safe_recipes)
    print("Target:", {k: round(v,1) for k,v in target.items()})
    
    # 3. Rank
    recs = rank_recipes(safe_recipes, target, params, slot, history)
    if not recs.empty:
        print("\nTop Recommendations:")
        print(recs[['title', 'calories', 'spiciness_score', 'score']].to_string(index=False))
        
    # 4. Train
    print("\nAdapting Parameters...")
    
    # Need to capture chosen recipe for report
    last = history.iloc[0]
    chosen = recipes[recipes['recipe_id'] == last['recipe_id']].iloc[0]
    
    old_params = json.loads(json.dumps(params)) # Deep copy for display
    new_params = train_weights(params, history, recipes, target)
    save_params(new_params)
    print("Done.")

    # 5. Report
    generate_math_report(profile, target, old_params, chosen, new_params)
