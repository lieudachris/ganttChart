"""
WBS Gantt Chart Generator with Givers, Dependencies, and Milestones
====================================================================
This script creates a professional Gantt chart from a CSV file with WBS hierarchy,
task dependencies, milestone highlighting, and giver/team identification.

Features:
- Hierarchical WBS structure display
- Task color-coding by giver/team
- Dependency arrows with right-angle paths
- Milestone diamond markers
- Professional legend and styling

Author: Data Visualization Expert
Date: October 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, AutoDateLocator
import numpy as np
import io
import os
from datetime import datetime, timedelta


# ============================================================================
# SAMPLE DATA - Replace with your own CSV file path
# ============================================================================
SAMPLE_CSV_DATA = """WBS ID,TaskName,Start Date,End Date,Duration,Predecessor,Milestone,Giver
1,Project Planning,2025-01-01,2025-01-15,,0,0,PM Team
1.1,Requirements Gathering,2025-01-01,2025-01-08,,1,0,PM Team
1.2,Risk Assessment,2025-01-08,2025-01-15,7,1.1,0,QA Lead
2,Design Phase,2025-01-15,2025-02-28,,1.2,0,Design Team
2.1,Architecture Design,2025-01-15,2025-01-31,,2,0,Arch Lead
2.2,Database Schema,2025-01-20,2025-02-10,22,2.1,0,DB Engineer
2.3,UI/UX Design,2025-01-25,2025-02-28,,2.1,1,Design Team
3,Development,2025-02-15,2025-04-30,,2.2,0,Dev Team
3.1,Backend Development,2025-02-15,2025-03-31,,2.2,0,Backend Lead
3.2,Frontend Development,2025-02-20,2025-04-15,,2.3,0,Frontend Lead
3.3,API Integration,2025-03-15,2025-04-10,,3.1;3.2,0,Dev Team
4,Testing,2025-04-01,2025-05-15,,3.1,0,QA Team
4.1,Unit Testing,2025-04-01,2025-04-20,,3.1,0,QA Engineer
4.2,Integration Testing,2025-04-20,2025-05-05,,4.1,0,QA Lead
4.3,UAT,2025-05-05,2025-05-15,,4.2,1,QA Team
5,Deployment,2025-05-15,2025-05-30,,4.3,0,DevOps Team
5.1,Production Setup,2025-05-15,2025-05-20,,4.3,0,DevOps Team
5.2,Deployment,2025-05-20,2025-05-25,,5.1,0,DevOps Team
5.3,Go-Live,2025-05-25,2025-05-30,,5.2,1,PM Team"""


# ============================================================================
# DATA LOADING AND PARSING
# ============================================================================
def load_data(csv_path=None):
    """
    Load CSV data from file or use sample data.
    Handles date parsing, WBS hierarchy level calculation, and giver assignment.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV_DATA))
    
    # Ensure WBS ID is string type
    df['WBS ID'] = df['WBS ID'].astype(str)
    
    # Parse dates
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    
    # Calculate missing Duration values
    df['Duration'] = df.apply(
        lambda row: (row['End Date'] - row['Start Date']).days 
        if pd.isna(row['Duration']) or row['Duration'] == '' else int(row['Duration']),
        axis=1
    )
    
    # Calculate WBS hierarchy level (number of dots + 1)
    df['Level'] = df['WBS ID'].apply(lambda x: len(str(x).split('.')) - 1)
    
    # Parse milestone column (accepts TRUE, True, Yes, yes, 1)
    # Convert to native Python bool to avoid numpy.bool_ issues
    df['Milestone'] = df['Milestone'].apply(
        lambda x: bool(x in [1, '1', 'True', 'TRUE', 'Yes', 'YES', 'yes', True])
    )
    
    # Clean up Predecessor column - handle empty values and whitespace
    df['Predecessor'] = df['Predecessor'].fillna('').astype(str).str.strip()
    
    # Ensure Giver column exists and is populated
    if 'Giver' not in df.columns:
        df['Giver'] = 'Unassigned'
    df['Giver'] = df['Giver'].fillna('Unassigned').astype(str)
    
    # Sort by WBS ID hierarchy first (to maintain parent-child relationships),
    # then by Start Date (to show chronological progression)
    df['WBS_sort'] = df['WBS ID'].str.split('.').apply(
        lambda x: tuple(map(int, x))
    )
    df = df.sort_values(['WBS_sort', 'Start Date']).reset_index(drop=True)
    df = df.drop('WBS_sort', axis=1)  # Clean up temporary column
    
    return df


# ============================================================================
# GIVER AND COLOR ASSIGNMENT
# ============================================================================
def get_giver_colors(givers):
    """
    Generate a distinct color palette for different givers/teams.
    Uses vibrant, contrasting colors for clear differentiation.
    """
    color_palette = [
        '#FF6B6B',  # Red
        '#45B7D1',  # Blue
        '#2ECC71',  # Green
        '#F39C12',  # Orange
        '#9B59B6',  # Purple
        '#1ABC9C',  # Teal
        '#E67E22',  # Burnt Orange
        '#34495E',  # Dark Gray
        '#E74C3C',  # Crimson
        '#3498DB',  # Sky Blue
        '#16A085',  # Sea Green
        '#8E44AD',  # Violet
    ]
    
    giver_color_map = {}
    for idx, giver in enumerate(sorted(set(givers))):
        giver_color_map[giver] = color_palette[idx % len(color_palette)]
    
    return giver_color_map


# ============================================================================
# GANTT CHART GENERATION
# ============================================================================
def generate_gantt_chart(df, output_path='charts/gantt_chart.png'):
    """
    Generate a comprehensive Gantt chart with:
    - Tasks colored by giver/team
    - Dependency arrows between tasks (right-angle paths)
    - Milestone diamond markers
    - WBS hierarchy labels
    """
    
    # Create charts folder if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-muted')
    except OSError:
        plt.style.use('default')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(18, max(10, len(df) * 0.4)))
    
    # Generate giver color mapping
    giver_color_map = get_giver_colors(df['Giver'])
    
    # Sort by Start Date for display (earliest at top)
    df_sorted = df.sort_values('Start Date').reset_index(drop=True)
    
    # Y-axis positions for tasks (earliest at top = highest y value, latest at bottom = lowest)
    y_positions = {idx: len(df_sorted) - 1 - pos for pos, idx in enumerate(df_sorted.index)}
    
    # Draw task bars
    for pos, (idx, row) in enumerate(df_sorted.iterrows()):
        y = y_positions[idx]
        start = row['Start Date']
        duration = (row['End Date'] - start).days
        
        # Get color based on giver
        color = giver_color_map[row['Giver']]
        
        # Draw task bar
        ax.barh(y, duration, left=start, height=0.6, 
                color=color, edgecolor='black', linewidth=1.5, alpha=0.85)
        
        # Add milestone diamond marker if applicable
        if row['Milestone']:
            mid_x = start + timedelta(days=duration/2)
            ax.plot(mid_x, y, marker='D', markersize=12, 
                   color='gold', markeredgecolor='darkgoldenrod', 
                   markeredgewidth=2.5, zorder=5)
    
    # ========================================================================
    # DRAW DEPENDENCY ARROWS (supports multiple predecessors per task)
    # Right-angle arrows with bends for cleaner visualization
    # ========================================================================
    for pos, (idx, row) in enumerate(df_sorted.iterrows()):
        predecessor_str = row['Predecessor']
        
        # Skip if no predecessors
        if not predecessor_str or predecessor_str == '' or pd.isna(predecessor_str):
            continue
        
        current_y = y_positions[idx]
        current_start = row['Start Date']
        
        # Split multiple predecessors by semicolon and process each one
        predecessors = [p.strip() for p in str(predecessor_str).split(';') if p.strip()]
        
        for pred_id in predecessors:
            # Find predecessor in original dataframe by WBS ID
            pred_rows = df[df['WBS ID'] == pred_id]
            if pred_rows.empty:
                continue
            
            pred_idx = pred_rows.index[0]
            pred_y = y_positions[pred_idx]
            pred_end = df.loc[pred_idx, 'End Date']
            
            # Create right-angle path: horizontal from pred_end, then vertical to current task
            # Use a shorter horizontal offset to avoid overlapping with other tasks
            offset_days = 0.5
            mid_date = pred_end + timedelta(days=offset_days)
            
            # Draw horizontal line from predecessor end
            ax.plot([pred_end, mid_date], [pred_y, pred_y], 
                   color='red', lw=1.2, alpha=0.6)
            
            # Draw vertical line down to current task
            ax.plot([mid_date, mid_date], [pred_y, current_y], 
                   color='red', lw=1.2, alpha=0.6)
            
            # Draw final horizontal line to current task start
            ax.plot([mid_date, current_start], [current_y, current_y], 
                   color='red', lw=1.2, alpha=0.6)
            
            # Draw arrowhead at the end
            ax.annotate('', xy=(current_start, current_y), 
                       xytext=(current_start - timedelta(days=0.3), current_y),
                       arrowprops=dict(arrowstyle='->', lw=1.2, 
                                     color='red', alpha=0.6))
    
    # ========================================================================
    # CONFIGURE AXES
    # ========================================================================
    ax.set_yticks(range(len(df_sorted)))
    
    # Create y-axis labels with WBS ID, Task Name, and Giver
    y_labels = [f"{row['WBS ID']} • {row['TaskName']} • 👤 {row['Giver']}" 
                for _, row in df_sorted.iloc[::-1].iterrows()]
    ax.set_yticklabels(y_labels, fontsize=9)
    
    ax.set_xlabel('Timeline (Date)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task • Giver', fontsize=12, fontweight='bold')
    ax.set_title('Project Gantt Chart with WBS Hierarchy, Givers, and Dependencies', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Format X-axis dates with adaptive locator
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45, ha='right')
    
    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # ========================================================================
    # CREATE LEGEND
    # ========================================================================
    legend_elements = []
    
    # Add giver colors to legend
    for giver in sorted(giver_color_map.keys()):
        color = giver_color_map[giver]
        legend_elements.append(mpatches.Patch(
            facecolor=color, edgecolor='black', linewidth=1.5,
            label=f'👤 {giver}'
        ))
    
    # Add milestone marker
    legend_elements.append(mpatches.Patch(
        facecolor='none', edgecolor='none', 
        label='◆ Milestone (Diamond Marker)'
    ))
    
    # Add dependency arrow
    legend_elements.append(mpatches.Patch(
        facecolor='none', edgecolor='red', 
        label='→ Task Dependency (Right-Angle)'
    ))
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.95, 
             title='Givers & Legend', title_fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gantt chart saved to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY AND STATISTICS
# ============================================================================
def print_project_summary(df):
    """
    Print a summary of project statistics including giver workload.
    """
    print("\n" + "="*70)
    print("PROJECT SUMMARY")
    print("="*70)
    print(f"Total Tasks: {len(df)}")
    print(f"Project Start: {df['Start Date'].min().strftime('%Y-%m-%d')}")
    print(f"Project End: {df['End Date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Duration: {(df['End Date'].max() - df['Start Date'].min()).days} days")
    print(f"\nWBS Hierarchy Levels: 0-{df['Level'].max()}")
    print(f"Milestones: {df['Milestone'].sum()}")
    
    print("\nGIVER WORKLOAD:")
    print("-" * 70)
    giver_stats = df.groupby('Giver').agg({
        'TaskName': 'count',
        'Duration': 'sum'
    }).rename(columns={'TaskName': 'Tasks', 'Duration': 'Total Days'})
    giver_stats = giver_stats.sort_values('Tasks', ascending=False)
    
    for giver, row in giver_stats.iterrows():
        print(f"  {giver:20} | Tasks: {int(row['Tasks']):3} | Duration: {int(row['Total Days']):3} days")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    # Uncomment the line below to use an external CSV file:
    csv_file_path = 'project_schedule.csv'
    # csv_file_path = None  # Use sample data
    
    # Load data
    df = load_data(csv_file_path)
    
    print("\n✓ Data loaded successfully!")
    print(f"  Total tasks: {len(df)}")
    print(f"  WBS hierarchy levels: 0-{df['Level'].max()}")
    print(f"  Unique givers: {df['Giver'].nunique()}")
    print(f"  Givers: {', '.join(sorted(df['Giver'].unique()))}")
    
    # Print project summary with giver statistics
    print_project_summary(df)
    
    # Generate Gantt chart
    timestamp = datetime.now().strftime('%Y%m%d')
    output_chart_path = f'charts/gantt_chart_{timestamp}.png'
    generate_gantt_chart(df, output_chart_path)
    
    print("✓ Chart generation complete!")
    print(f"  Output: {output_chart_path}\n")


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
HOW TO USE THIS SCRIPT:
=======================

1. USING SAMPLE DATA (default):
   - Simply run the script: python gantt_chart.py
   - A Gantt chart will be generated in charts/gantt_chart_YYYYMMDD.png
   - Project summary with giver statistics will be printed

2. USING YOUR OWN CSV FILE:
   - Place your CSV file in the project directory
   - Uncomment this line in the main section:
     csv_file_path = 'your_file.csv'
   - Run the script

3. CSV FILE FORMAT:
   Required columns:
   - WBS ID: Hierarchical identifier (e.g., 1, 1.1, 1.1.1)
   - TaskName: Name of the task
   - Start Date: YYYY-MM-DD format
   - End Date: YYYY-MM-DD format
   - Duration: Days (optional, auto-calculated from dates)
   - Predecessor: WBS ID(s) of predecessor task(s), separated by semicolon (;)
   - Milestone: TRUE/Yes/1 for milestones, FALSE/No/0 otherwise
   - Giver: Name of person/team responsible for the task

4. GIVER FEATURES:
   - Each giver is assigned a distinct color
   - Y-axis labels show: WBS • Task • 👤 Giver
   - Legend displays all givers with their color mapping
   - Project summary shows workload per giver (task count and total days)
   - Use 'Unassigned' if Giver field is empty

5. OUTPUT:
   - Charts are saved to charts/gantt_chart_YYYYMMDD.png
   - Each run creates a new file with today's date
   - Console output shows project statistics and giver workload

FEATURES:
==========
✓ WBS hierarchy visualization with chronological ordering
✓ Color-coded tasks by giver/team for workload visibility
✓ Dependency arrows with right-angle paths
✓ Milestone diamond markers
✓ Professional Matplotlib styling
✓ Multi-predecessor support (semicolon-separated)
✓ Auto-calculated task duration
✓ Giver workload statistics
✓ No GUI - lightweight, fast execution
✓ Easy CSV integration
"""