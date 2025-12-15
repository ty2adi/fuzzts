import pandas as pd
from datetime import datetime, timedelta
import random

def generate_mock_input_table(recipes_df, user_id="user_123", days=10):
    """
    Generates a mock history of 'past orders' for a user.
    Simulates ~3 meals a day (Breakfast, Lunch, Dinner).
    
    Args:
        recipes_df: DataFrame of available recipes to pick from.
        user_id: ID of the user.
        days: Number of past days to simulate.
        
    Returns:
        pd.DataFrame: Columns [user_id, date, meal_slot, recipe_id]
    """
    history = []
    end_date = datetime.now()
    
    # Filter recipes by type so we pick valid meals for history slots
    breakfast_pool = recipes_df[recipes_df['meal_type'] == 'breakfast']['recipe_id'].tolist()
    lunch_pool = recipes_df[recipes_df['meal_type'] == 'lunch']['recipe_id'].tolist()
    dinner_pool = recipes_df[recipes_df['meal_type'] == 'dinner']['recipe_id'].tolist()
    
    if not (breakfast_pool and lunch_pool and dinner_pool):
        raise ValueError("Recipe dataset missing meals for some slots.")

    for day_offset in range(days):
        date = end_date - timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        
        # Add Breakfast
        if breakfast_pool:
            history.append({
                "user_id": user_id,
                "date": f"{date_str} 08:30:00",
                "meal_slot": "Breakfast",
                "recipe_id": random.choice(breakfast_pool)
            })
            
        # Add Lunch
        if lunch_pool:
            history.append({
                "user_id": user_id,
                "date": f"{date_str} 13:00:00",
                "meal_slot": "Lunch",
                "recipe_id": random.choice(lunch_pool)
            })
            
        # Add Dinner
        if dinner_pool:
            history.append({
                "user_id": user_id,
                "date": f"{date_str} 19:30:00",
                "meal_slot": "Dinner",
                "recipe_id": random.choice(dinner_pool)
            })
            
    # Convert to DF and sort by date descending (latest first)
    history_df = pd.DataFrame(history)
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df = history_df.sort_values(by='date', ascending=False).reset_index(drop=True)
    
    return history_df

if __name__ == "__main__":
    # Test
    try:
        df = pd.read_csv("recipes_rows_spice0_100_minimal.csv")
        history = generate_mock_input_table(df)
        print("Mock Input Table (Top 5 rows):")
        print(history.head())
        print(f"\nTotal History Records: {len(history)}")
        
        # Save to CSV for POC persistence
        history.to_csv("mock_input_table.csv", index=False)
        print("Saved to mock_input_table.csv")
    except Exception as e:
        print(f"Error: {e}")
