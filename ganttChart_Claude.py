"""
WBS Gantt Chart Generator with Dependencies, Milestones, and Interactive GUI
==============================================================================
This script creates a professional Gantt chart from a CSV file with WBS hierarchy,
task dependencies, and milestone highlighting. It includes a Tkinter GUI for 
editing task properties and redrawing the chart.

Author: Data Visualization Expert
Date: October 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, MonthLocator
import numpy as np
import io
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading


# ============================================================================
# SAMPLE DATA - Replace with your own CSV file path
# ============================================================================
SAMPLE_CSV_DATA = """WBS ID,TaskName,Start Date,End Date,Duration,Predecessor,Milestone
1,Project Planning,2025-01-01,2025-01-15,,0
1.1,Requirements Gathering,2025-01-01,2025-01-08,,1,0
1.2,Risk Assessment,2025-01-08,2025-01-15,7,1.1,0
2,Design Phase,2025-01-15,2025-02-28,,1.2,0
2.1,Architecture Design,2025-01-15,2025-01-31,,2,0
2.2,Database Schema,2025-01-20,2025-02-10,22,2.1,0
2.3,UI/UX Design,2025-01-25,2025-02-28,,2.1,1
3,Development,2025-02-15,2025-04-30,,2.2,0
3.1,Backend Development,2025-02-15,2025-03-31,,2.2,0
3.2,Frontend Development,2025-02-20,2025-04-15,,2.3,0
3.3,API Integration,2025-03-15,2025-04-10,,3.1;3.2,0
4,Testing,2025-04-01,2025-05-15,,3.1,0
4.1,Unit Testing,2025-04-01,2025-04-20,,3.1,0
4.2,Integration Testing,2025-04-20,2025-05-05,,4.1,0
4.3,UAT,2025-05-05,2025-05-15,,4.2,1
5,Deployment,2025-05-15,2025-05-30,,4.3,0
5.1,Production Setup,2025-05-15,2025-05-20,,4.3,0
5.2,Deployment,2025-05-20,2025-05-25,,5.1,0
5.3,Go-Live,2025-05-25,2025-05-30,,5.2,1"""


# ============================================================================
# DATA LOADING AND PARSING
# ============================================================================
def load_data(csv_path=None):
    """
    Load CSV data from file or use sample data.
    Handles date parsing and WBS hierarchy level calculation.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV_DATA))
    
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
    # Convert to native Python bool to avoid numpy.bool_ issues with Tkinter
    df['Milestone'] = df['Milestone'].apply(
        lambda x: bool(x in [1, '1', 'True', 'TRUE', 'Yes', 'YES', 'yes', True])
    )
    
    # Clean up Predecessor column - handle empty values and whitespace
    df['Predecessor'] = df['Predecessor'].fillna('').astype(str).str.strip()
    
    # Sort by WBS ID to maintain hierarchy order
    df = df.sort_values('WBS ID', key=lambda x: x.str.split('.').apply(
        lambda y: tuple(map(int, y))
    )).reset_index(drop=True)
    
    return df


# ============================================================================
# HIERARCHY AND COLOR ASSIGNMENT
# ============================================================================
def get_hierarchy_colors(max_level):
    """
    Generate a color palette for different WBS hierarchy levels.
    Each level has a distinctly different color for clear differentiation.
    """
    colors = [
        '#e74c3c',  # Level 0: Bold red
        '#3498db',  # Level 1: Bold blue
        '#2ecc71',  # Level 2: Bold green
        '#f39c12',  # Level 3: Bold orange
        '#9b59b6',  # Level 4: Bold purple
        '#1abc9c',  # Level 5: Bold teal
        '#e67e22',  # Level 6: Bold burnt orange
        '#34495e',  # Level 7: Bold dark gray
    ]
    return colors[:max_level + 1] + ['#95a5a6'] * (max_level - len(colors) + 2)


def get_color_for_level(level, colors):
    """Get color for a specific hierarchy level."""
    return colors[min(level, len(colors) - 1)]


# ============================================================================
# GANTT CHART GENERATION
# ============================================================================
def generate_gantt_chart(df, output_path='charts/gantt_chart.png'):
    """
    Generate a comprehensive Gantt chart with:
    - Tasks colored by WBS hierarchy level
    - Dependency arrows between tasks
    - Milestone diamond markers
    """
    
    # Create charts folder if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-muted')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.35)))
    
    # Generate colors
    colors = get_hierarchy_colors(df['Level'].max())
    
    # Sort by WBS ID hierarchy first (to maintain parent-child relationships),
    # then by Start Date (to show chronological progression)
    df['WBS_sort'] = df['WBS ID'].str.split('.').apply(
        lambda x: tuple(map(int, x))
    )
    df_sorted = df.sort_values(['WBS_sort', 'Start Date']).reset_index(drop=True)
    df = df.drop('WBS_sort', axis=1)  # Clean up temporary column
    
    # Y-axis positions for tasks (earliest at top = highest y value, latest at bottom = lowest)
    y_positions = {idx: len(df_sorted) - 1 - pos for pos, idx in enumerate(df_sorted.index)}
    
    # Draw task bars
    for pos, (idx, row) in enumerate(df_sorted.iterrows()):
        y = y_positions[idx]
        start = row['Start Date']
        duration = (row['End Date'] - start).days
        
        # Get color based on hierarchy level
        color = get_color_for_level(row['Level'], colors)
        
        # Draw task bar
        ax.barh(y, duration, left=start, height=0.6, 
                color=color, edgecolor='black', linewidth=1.5)
        
        # Add milestone diamond marker if applicable
        if row['Milestone']:
            mid_x = start + timedelta(days=duration/2)
            ax.plot(mid_x, y, marker='D', markersize=10, 
                   color='gold', markeredgecolor='darkgoldenrod', 
                   markeredgewidth=2, zorder=5)
    
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
            offset_days = 0.5  # Reduced from 2 days for tighter arrows
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
    ax.set_yticks(range(len(df)))
    # Create y-axis labels with both WBS ID and Task Name
    # Reverse the order to match the bar positions (earliest at top)
    y_labels = [f"{row['WBS ID']} - {row['TaskName']}" 
                for _, row in df_sorted.iloc[::-1].iterrows()]
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Timeline (Date)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Task Name', fontsize=11, fontweight='bold')
    ax.set_title('ESPAStar-D FMSA Software Release Flow', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Format X-axis dates with adaptive locator based on date range
    from matplotlib.dates import AutoDateLocator
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
    
    # Add hierarchy level colors
    for level in range(df['Level'].max() + 1):
        color = get_color_for_level(level, colors)
        legend_elements.append(mpatches.Patch(
            facecolor=color, edgecolor='black', 
            label=f'WBS Level {level}'
        ))
    
    # Add milestone marker
    legend_elements.append(mpatches.Patch(
        facecolor='none', edgecolor='none', 
        label='◆ Milestone (Diamond Marker)'
    ))
    
    # Add dependency arrow
    legend_elements.append(mpatches.Patch(
        facecolor='none', edgecolor='red', 
        label='→ Task Dependency'
    ))
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gantt chart saved to {output_path}")
    plt.close()


# ============================================================================
# GUI INTERFACE WITH TKINTER
# ============================================================================
class GanttChartGUI:
    """
    Interactive GUI for editing task properties and regenerating the Gantt chart.
    """
    
    def __init__(self, root, csv_path=None):
        self.root = root
        self.root.title("WBS Gantt Chart Manager")
        self.root.geometry("700x500")
        
        # Load initial data
        self.csv_path = csv_path
        self.df = load_data(csv_path)
        self.original_df = self.df.copy()
        
        # Create GUI elements
        self.create_widgets()
        self.update_task_dropdown()
        
    def create_widgets(self):
        """Create GUI layout with dropdown and editing fields."""
        
        # Frame for task selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(top_frame, text="Select Task:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        
        self.task_dropdown = ttk.Combobox(top_frame, state='readonly', width=40)
        self.task_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.task_dropdown.bind('<<ComboboxSelected>>', self.on_task_selected)
        
        # Frame for editing fields
        edit_frame = ttk.LabelFrame(self.root, text="Edit Task Properties", padding="10")
        edit_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                       padx=10, pady=10)
        
        # Start Date
        ttk.Label(edit_frame, text="Start Date (YYYY-MM-DD):").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.start_date_entry = ttk.Entry(edit_frame, width=20)
        self.start_date_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # End Date
        ttk.Label(edit_frame, text="End Date (YYYY-MM-DD):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.end_date_entry = ttk.Entry(edit_frame, width=20)
        self.end_date_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Duration (read-only, auto-calculated)
        ttk.Label(edit_frame, text="Duration (days, auto-calculated):").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.duration_label = ttk.Label(edit_frame, text="", font=('Arial', 9))
        self.duration_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Milestone checkbox
        ttk.Label(edit_frame, text="Milestone:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.milestone_var = tk.BooleanVar()
        self.milestone_check = ttk.Checkbutton(edit_frame, variable=self.milestone_var)
        self.milestone_check.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.apply_button = ttk.Button(button_frame, text="Apply Changes", 
                                       command=self.apply_changes)
        self.apply_button.pack(side=tk.LEFT, padx=5)
        
        self.redraw_button = ttk.Button(button_frame, text="Redraw Chart", 
                                        command=self.redraw_chart)
        self.redraw_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(button_frame, text="Reset to Original", 
                                       command=self.reset_data)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready", 
                                      foreground="green", font=('Arial', 9))
        self.status_label.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
    
    def update_task_dropdown(self):
        """Populate dropdown with task names."""
        self.task_dropdown['values'] = list(self.df['TaskName'])
        if len(self.df) > 0:
            self.task_dropdown.current(0)
            self.on_task_selected(None)
    
    def on_task_selected(self, event):
        """Handle task selection and populate edit fields."""
        selected_task = self.task_dropdown.get()
        task_row = self.df[self.df['TaskName'] == selected_task]
        
        if task_row.empty:
            return
        
        task_data = task_row.iloc[0]
        self.start_date_entry.delete(0, tk.END)
        self.start_date_entry.insert(0, task_data['Start Date'].strftime('%Y-%m-%d'))
        
        self.end_date_entry.delete(0, tk.END)
        self.end_date_entry.insert(0, task_data['End Date'].strftime('%Y-%m-%d'))
        
        self.milestone_var.set(task_data['Milestone'])
        self.update_duration_display()
    
    def update_duration_display(self):
        """Calculate and display duration based on dates."""
        try:
            start = pd.to_datetime(self.start_date_entry.get())
            end = pd.to_datetime(self.end_date_entry.get())
            duration = (end - start).days
            self.duration_label.config(text=f"{duration} days")
        except:
            self.duration_label.config(text="Invalid dates")
    
    def apply_changes(self):
        """Apply changes to selected task and update dataframe."""
        selected_task = self.task_dropdown.get()
        
        try:
            start_date = pd.to_datetime(self.start_date_entry.get())
            end_date = pd.to_datetime(self.end_date_entry.get())
            
            if start_date >= end_date:
                messagebox.showerror("Error", "Start Date must be before End Date")
                return
            
            # Update dataframe
            mask = self.df['TaskName'] == selected_task
            self.df.loc[mask, 'Start Date'] = start_date
            self.df.loc[mask, 'End Date'] = end_date
            self.df.loc[mask, 'Duration'] = (end_date - start_date).days
            self.df.loc[mask, 'Milestone'] = self.milestone_var.get()
            
            self.status_label.config(text="Changes applied successfully!", 
                                    foreground="green")
            self.update_duration_display()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
            self.status_label.config(text="Error applying changes", 
                                    foreground="red")
    
    def redraw_chart(self):
        """Redraw the Gantt chart with current data."""
        try:
            # Generate timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d')
            output_path = f'charts/gantt_chart_{timestamp}.png'
            
            # Run chart generation in separate thread to avoid GUI freeze
            thread = threading.Thread(target=generate_gantt_chart, 
                                     args=(self.df, output_path))
            thread.start()
            
            self.status_label.config(text=f"Chart saved to {output_path}", 
                                    foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to redraw chart: {str(e)}")
            self.status_label.config(text="Error redrawing chart", 
                                    foreground="red")
    
    def reset_data(self):
        """Reset dataframe to original state."""
        self.df = self.original_df.copy()
        self.update_task_dropdown()
        self.status_label.config(text="Data reset to original", 
                                foreground="green")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    # Uncomment the line below to use an external CSV file:
    csv_file_path = 'project_schedule.csv'
    # csv_file_path = None  # Use sample data
    
    # Load data
    df = load_data(csv_file_path)
    
    print("Data loaded successfully!")
    print(f"Total tasks: {len(df)}")
    print(f"WBS hierarchy levels: 0-{df['Level'].max()}")
    
    # Generate initial Gantt chart
    timestamp = datetime.now().strftime('%Y%m%d')
    output_chart_path = f'charts/gantt_chart_{timestamp}.png'
    generate_gantt_chart(df, output_chart_path)
    
    # # Launch GUI
    # root = tk.Tk()
    # gui = GanttChartGUI(root, csv_file_path)
    # root.mainloop()


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
HOW TO USE THIS SCRIPT:
=======================

1. USING SAMPLE DATA (default):
   - Simply run the script: python gantt_chart.py
   - A Gantt chart will be generated in charts/gantt_chart_YYYYMMDD.png
   - A GUI window will open for editing task properties

2. USING YOUR OWN CSV FILE:
   - Place your CSV file in the project directory
   - Uncomment this line in the main section:
     csv_file_path = 'your_file.csv'
   - Run the script

3. CSV FILE FORMAT:
   Required columns: WBS ID, TaskName, Start Date, End Date, Duration, Predecessor, Milestone
   - WBS ID: Hierarchical identifier (e.g., 1, 1.1, 1.1.1)
   - TaskName: Name of the task
   - Start Date: YYYY-MM-DD format
   - End Date: YYYY-MM-DD format
   - Duration: Days (optional, auto-calculated from dates)
   - Predecessor: WBS ID(s) of predecessor task(s), separated by semicolon (;)
   - Milestone: TRUE/Yes/1 for milestones, FALSE/No/0 otherwise

4. USING THE GUI:
   - Select a task from the dropdown
   - Edit Start Date, End Date, and Milestone status
   - Click "Apply Changes" to update the task
   - Click "Redraw Chart" to regenerate the Gantt chart with new dates
   - Click "Reset to Original" to undo all changes

5. OUTPUT:
   - Charts are saved to charts/gantt_chart_YYYYMMDD.png
   - Each redraw creates a new file with today's date
   - Original sample chart is generated on first run

FEATURES:
==========
✓ WBS hierarchy visualization with color-coded levels
✓ Dependency arrows showing task relationships
✓ Milestone diamond markers
✓ Interactive GUI for task editing
✓ Automatic chart regeneration
✓ Professional Matplotlib styling
✓ Multi-predecessor support (semicolon-separated)
✓ Auto-calculated task duration
"""