"""
Recipe Repetition Visualizer - Data-Driven Analysis
Reads simulation_report.txt and generates interactive visualizations
showing how score, penalty, and distance interact for the most repeated recipe.
"""

import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np


def parse_simulation_report(filepath="simulation_report.txt"):
    """
    Parse the simulation report to extract:
    - Day-by-day selections
    - Scores, distances, penalties for all candidates
    - Recipe names and IDs
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding which accepts all byte values
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Extract data structures
    days_data = []
    current_day = None
    current_slot = None
    current_target = {}
    
    # Regex patterns
    day_pattern = r"--- DAY (\d+): ([\d-]+) ---"
    slot_pattern = r"\[(\w+)\] Target: Cal=(\d+), Spice=(\d+)"
    recipe_pattern = r"#(\d+) \[(SELECTED|REJECTED)\]: (.+)"
    metrics_pattern = r"Sim: ([\d.]+) \| Match\(Dist\^2\): ([\d.]+) \| Penalty: ([\d.]+)"
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Match day header
        day_match = re.match(day_pattern, line)
        if day_match:
            current_day = int(day_match.group(1))
            current_date = day_match.group(2)
            i += 1
            continue
        
        # Match meal slot
        slot_match = re.match(slot_pattern, line)
        if slot_match:
            current_slot = slot_match.group(1)
            current_target = {
                'calories': int(slot_match.group(2)),
                'spice': int(slot_match.group(3))
            }
            i += 1
            continue
        
        # Match recipe entry
        recipe_match = re.match(recipe_pattern, line)
        if recipe_match:
            rank = int(recipe_match.group(1))
            status = recipe_match.group(2)
            recipe_name = recipe_match.group(3)
            
            # Next line should have metrics
            i += 1
            if i < len(lines):
                metrics_line = lines[i].strip()
                metrics_match = re.search(metrics_pattern, metrics_line)
                
                if metrics_match:
                    similarity = float(metrics_match.group(1))
                    distance = float(metrics_match.group(2))
                    penalty = float(metrics_match.group(3))
                    
                    days_data.append({
                        'day': current_day,
                        'slot': current_slot,
                        'rank': rank,
                        'status': status,
                        'recipe_name': recipe_name,
                        'similarity': similarity,
                        'distance': distance,
                        'penalty': penalty,
                        'target_cal': current_target['calories'],
                        'target_spice': current_target['spice']
                    })
        
        i += 1
    
    return pd.DataFrame(days_data)


def parse_continuous_tracking(filepath="simulation_report.txt"):
    """
    Parse the continuous tracking section at the end of the report.
    Returns tuple: (DataFrame with day-by-day data, recipe_name)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Find continuous tracking section
    if "CONTINUOUS TRACKING:" not in content:
        return None, None
    
    tracking_section = content.split("CONTINUOUS TRACKING:")[1]
    lines = tracking_section.split('\n')
    
    # Extract recipe name from the section
    recipe_name = None
    for line in lines:
        if line.startswith("Recipe:"):
            # Format: "Recipe: Green Thai Curry with Beef (ID: 586)"
            recipe_name = line.split("Recipe:")[1].split("(ID:")[0].strip()
            break
    
    tracking_data = []
    for line in lines:
        # Parse format: "Day | Distance(D²) | Penalty(P) | Score(S) | Status"
        # Example: "  1 |      48.5912 |     0.0000 |   0.020065 | SELECTED"
        if '|' in line and line.strip() and not line.startswith('-') and not line.startswith('Format'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 5:
                try:
                    day = int(parts[0])
                    distance = float(parts[1])
                    penalty = float(parts[2])
                    score = float(parts[3])
                    status = parts[4]
                    
                    tracking_data.append({
                        'day': day,
                        'distance': distance,
                        'penalty': penalty,
                        'score': score,
                        'status': status
                    })
                except (ValueError, IndexError):
                    continue
    
    df = pd.DataFrame(tracking_data) if tracking_data else None
    return df, recipe_name


def find_most_repeated_recipe(df):
    """
    Find the recipe that was selected most frequently.
    Returns recipe name, count, and slot breakdown.
    """
    selected = df[df['status'] == 'SELECTED']
    recipe_counts = Counter(selected['recipe_name'])
    
    if not recipe_counts:
        return None, 0, {}
    
    most_repeated = recipe_counts.most_common(1)[0]
    recipe_name = most_repeated[0]
    count = most_repeated[1]
    
    # Get slot breakdown
    slot_breakdown = selected[selected['recipe_name'] == recipe_name]['slot'].value_counts().to_dict()
    
    return recipe_name, count, slot_breakdown


def get_recipe_timeline(df, recipe_name):
    """
    Get the complete timeline for a specific recipe showing:
    - Days it was selected
    - Days it appeared in top 5 but wasn't selected
    - Score, distance, penalty evolution
    """
    recipe_data = df[df['recipe_name'] == recipe_name].copy()
    recipe_data = recipe_data.sort_values(['day', 'slot'])
    
    # Get selection days
    selected_days = recipe_data[recipe_data['status'] == 'SELECTED']['day'].tolist()
    
    return recipe_data, selected_days


def calculate_alternative_scores(df, recipe_name):
    """
    For comparison, get the winning recipe's score on days when target recipe wasn't selected.
    """
    # Get days where our target recipe appeared but wasn't selected
    target_rejected = df[(df['recipe_name'] == recipe_name) & (df['status'] == 'REJECTED')]
    
    alternatives = []
    for _, row in target_rejected.iterrows():
        # Find what was selected instead
        selected = df[(df['day'] == row['day']) & 
                     (df['slot'] == row['slot']) & 
                     (df['status'] == 'SELECTED')]
        
        if not selected.empty:
            alternatives.append({
                'day': row['day'],
                'slot': row['slot'],
                'alt_name': selected.iloc[0]['recipe_name'],
                'alt_similarity': selected.iloc[0]['similarity'],
                'alt_distance': selected.iloc[0]['distance'],
                'target_similarity': row['similarity'],
                'target_penalty': row['penalty']
            })
    
    return pd.DataFrame(alternatives)


def create_visualization(df, recipe_name, recipe_data, selected_days, alternatives_df, continuous_data=None):
    """
    Create interactive Plotly visualization with multiple subplots.
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'Score Evolution: {recipe_name}',
            'Component Breakdown (Distance vs Penalty)',
            'Selection Pattern & Comparison'
        ),
        vertical_spacing=0.12,
        row_heights=[0.35, 0.35, 0.3]
    )
    
    
    # Use continuous data if available, otherwise fall back to sparse data
    if continuous_data is not None and not continuous_data.empty:
        days = continuous_data['day'].tolist()
        scores = continuous_data['score'].tolist()
        distances = continuous_data['distance'].tolist()
        penalties = continuous_data['penalty'].tolist()
        statuses = continuous_data['status'].tolist()
        
        # Color code by selection status
        colors = ['#10b981' if s == 'SELECTED' else '#667eea' for s in statuses]
        marker_sizes = [12 if s == 'SELECTED' else 8 for s in statuses]
    else:
        # Fallback to aggregated daily data
        daily_data = recipe_data.groupby('day').agg({
            'similarity': 'mean',
            'distance': 'mean',
            'penalty': 'mean',
            'status': lambda x: 'SELECTED' if 'SELECTED' in x.values else 'REJECTED'
        }).reset_index()
        
        days = daily_data['day'].tolist()
        scores = daily_data['similarity'].tolist()
        distances = daily_data['distance'].tolist()
        penalties = daily_data['penalty'].tolist()
        statuses = daily_data['status'].tolist()
        
        colors = ['#667eea' if s == 'SELECTED' else '#f59e0b' for s in statuses]
        marker_sizes = [10] * len(days)
    
    # Subplot 1: Score Evolution
    fig.add_trace(
        go.Scatter(
            x=days, y=scores,
            mode='lines+markers',
            name='Similarity Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
            hovertemplate='Day %{x}<br>Score: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add markers for selected days
    selected_scores = [scores[days.index(d)] for d in selected_days if d in days]
    fig.add_trace(
        go.Scatter(
            x=selected_days, y=selected_scores,
            mode='markers',
            name='Selected',
            marker=dict(size=15, color='#10b981', symbol='star', 
                       line=dict(width=2, color='white')),
            hovertemplate='Day %{x}<br>SELECTED<br>Score: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Subplot 2: Distance vs Penalty
    fig.add_trace(
        go.Bar(
            x=days, y=distances,
            name='Distance (D²)',
            marker_color='#ef4444',
            hovertemplate='Day %{x}<br>Distance: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=days, y=penalties,
            name='Penalty (P)',
            marker_color='#10b981',
            hovertemplate='Day %{x}<br>Penalty: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Subplot 3: Selection pattern timeline
    # Create binary selection indicator
    selection_indicator = [1 if d in selected_days else 0 for d in range(1, max(days)+1)]
    all_days = list(range(1, max(days)+1))
    
    fig.add_trace(
        go.Bar(
            x=all_days, y=selection_indicator,
            name='Selected',
            marker_color='#667eea',
            hovertemplate='Day %{x}<br>%{text}<extra></extra>',
            text=['SELECTED' if s == 1 else 'Not Selected' for s in selection_indicator]
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Day", row=3, col=1)
    fig.update_yaxes(title_text="Similarity Score", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Selected", row=3, col=1, tickvals=[0, 1], ticktext=['No', 'Yes'])
    
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text=f"Recipe Repetition Analysis: {recipe_name}",
        title_font_size=20,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def generate_insights(recipe_name, count, slot_breakdown, selected_days, recipe_data, alternatives_df):
    """
    Generate text-based insights about the repetition pattern.
    """
    print("\n" + "="*80)
    print(f"RECIPE REPETITION ANALYSIS: {recipe_name}")
    print("="*80)
    
    print(f"\n📊 REPETITION FREQUENCY")
    print(f"   Selected {count} times across {len(selected_days)} days")
    print(f"   Slot breakdown: {slot_breakdown}")
    
    # Calculate gaps between selections
    if len(selected_days) > 1:
        gaps = [selected_days[i] - selected_days[i-1] for i in range(1, len(selected_days))]
        avg_gap = np.mean(gaps)
        print(f"\n🔄 PERIODIC PATTERN")
        print(f"   Selected on days: {selected_days}")
        print(f"   Gaps between selections: {gaps}")
        print(f"   Average gap: {avg_gap:.1f} days")
        
        if max(gaps) <= 7:
            print(f"   ⚠️ WARNING: Periodic loop detected! Recipe repeats every ~{avg_gap:.0f} days")
    
    # Analyze why it repeats
    selected_data = recipe_data[recipe_data['status'] == 'SELECTED']
    if not selected_data.empty:
        avg_distance = selected_data['distance'].mean()
        avg_penalty_when_selected = selected_data['penalty'].mean()
        
        print(f"\n🎯 MATHEMATICAL ANALYSIS")
        print(f"   Average Distance (D²) when selected: {avg_distance:.2f}")
        print(f"   Average Penalty when selected: {avg_penalty_when_selected:.2f}")
        print(f"   Average Score when selected: {selected_data['similarity'].mean():.4f}")
    
    # Compare with alternatives
    if not alternatives_df.empty:
        print(f"\n💡 COMPARISON WITH ALTERNATIVES")
        print(f"   Times rejected in favor of other recipes: {len(alternatives_df)}")
        
        for _, row in alternatives_df.head(3).iterrows():
            print(f"   Day {row['day']} ({row['slot']}): Lost to '{row['alt_name']}'")
            print(f"      Target score: {row['target_similarity']:.4f} (penalty: {row['target_penalty']:.2f})")
            print(f"      Winner score: {row['alt_similarity']:.4f}")
    
    print("\n" + "="*80)


def main():
    """
    Main execution function.
    """
    print("🔍 Parsing simulation report...")
    df = parse_simulation_report("simulation_report.txt")
    
    if df.empty:
        print("❌ No data found in simulation report!")
        return
    
    print(f"✅ Loaded {len(df)} recipe entries across {df['day'].nunique()} days")
    
    # Find most repeated recipe
    recipe_name, count, slot_breakdown = find_most_repeated_recipe(df)
    
    if not recipe_name:
        print("❌ No selected recipes found!")
        return
    
    # Parse continuous tracking data if available
    print("\n📊 Parsing continuous tracking data...")
    continuous_data, tracked_recipe_name = parse_continuous_tracking("simulation_report.txt")
    
    if continuous_data is not None and tracked_recipe_name:
        print(f"✅ Found continuous tracking for {len(continuous_data)} days")
        print(f"   Tracked recipe: {tracked_recipe_name}")
        # Use the recipe from continuous tracking
        recipe_name = tracked_recipe_name
        count = len(continuous_data[continuous_data['status'] == 'SELECTED'])
        
        # Get timeline data for this recipe
        recipe_data, selected_days = get_recipe_timeline(df, recipe_name)
        
        # Get slot breakdown
        slot_breakdown = recipe_data[recipe_data['status'] == 'SELECTED']['slot'].value_counts().to_dict()
    else:
        # Fallback to parsing from day-by-day entries
        recipe_name, count, slot_breakdown = find_most_repeated_recipe(df)
        
        if not recipe_name:
            print("❌ No selected recipes found!")
            return
        
        # Get timeline data
        recipe_data, selected_days = get_recipe_timeline(df, recipe_name)
    
    print(f"\n🏆 Most repeated recipe: {recipe_name} ({count} times)")
    
    # Get alternatives
    alternatives_df = calculate_alternative_scores(df, recipe_name)
    
    # Generate insights
    generate_insights(recipe_name, count, slot_breakdown, selected_days, 
                     recipe_data, alternatives_df)
    
    # Create visualization
    print("\n📊 Generating interactive visualization...")
    fig = create_visualization(df, recipe_name, recipe_data, selected_days, alternatives_df, continuous_data)
    
    # Save to HTML
    output_file = "recipe_repetition_analysis.html"
    fig.write_html(output_file)
    print(f"✅ Visualization saved to: {output_file}")
    
    # Also show in browser
    fig.show()
    
    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
