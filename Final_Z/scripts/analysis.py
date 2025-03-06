import pandas as pd
import numpy as np
import datetime as dt

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.io import output_notebook
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from IPython.display import display
output_notebook()
import panel as pn
pn.extension()
import holoviews as hv
from ipywidgets import VBox
hv.extension('bokeh')

# Statistics libraries
from scipy import stats
from scipy.stats import norm, shapiro, mannwhitneyu
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import Output  # ✅ Corrected Import
from scipy.stats import anderson

######################################################################################################################################################
def miss_values(df):
    # Function to calculate the percentage of missing values in each column
    dic_miss_value = {}  # Initialize an empty dictionary to store missing values for each column
    
    # Loop through each column to calculate the missing values percentage
    for col in df.columns:
        num_miss = df[col].isna().sum()  # Count the number of missing values (NaN) in the column
        percentage_miss = num_miss / len(df) if len(df) > 0 else 0  # Avoid division by zero
        dic_miss_value[col] = percentage_miss  # Store the percentage of missing values
    
    # Calculate the percentage of missing values in the entire DataFrame
    num_miss_df = df.isna().sum().sum() / df.size if df.size > 0 else 0
    
    return dic_miss_value, num_miss_df
######################################################################################################################################################
def fill_clear(df, missing_dict):
    df_copy = df.copy()  # To avoid direct modification of the original DataFrame
    for key in missing_dict.keys():  
        # Check if the specified column exists in the DataFrame
        if key in df_copy.columns:
            # Only if the column is numeric, fill missing values with the mean
            if pd.api.types.is_numeric_dtype(df_copy[key]):
                mean_value = df_copy[key].mean()  # Calculate the mean
                df_copy[key] = df_copy[key].fillna(mean_value)  # Fill missing values with the mean
            else:
                print(f"Skipping column '{key}' as it is non-numeric.")
        else:
            print(f"Column '{key}' not found in DataFrame.")
    
    return df_copy
######################################################################################################################################################
def remove_outliers_iqr(df, columns):
    df_clean = df.copy() 
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        else:
            print(f"Skipping column '{col}' (not numeric or not found).")
    
    return df_clean

#####################################################################################################################################################
def plot_qq_plots(df, columns, labels):
    """
    Generates QQ plots for heart rate data of four participants.
    
    Parameters:
        df (DataFrame): The dataframe containing heart rate data.
        columns (list): List of column names for heart rate data.
        labels (list): List of corresponding participant labels.
    """
    plt.figure(figsize=(12, 8))  # Define figure size

    for i, (col, label) in enumerate(zip(columns, labels), 1):
        plt.subplot(2, 2, i)  # Arrange plots in a 2x2 grid
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)  # QQ Plot
        plt.title(f'QQ Plot for {label}')  # Title for each plot

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()  # Show the plots

######################################################################################################################################################
def plot_qq_plot(df, columns, labels, title="QQ Plot"):
    """
    Generates an interactive QQ plot for multiple participants.

    Parameters:
        df (DataFrame): The dataframe containing heart rate data.
        columns (list): List of column names for heart rate data.
        labels (list): List of corresponding participant labels.
        title (str): Title of the QQ plot.
    """
    fig = go.Figure()

    for col, label in zip(columns, labels):
        data = pd.to_numeric(df[col], errors='coerce').dropna()  # Convert to numeric and drop NaNs
        
        if data.empty:
            print(f"Warning: No valid numeric data for {label}, skipping.")
            continue  # Skip if no valid data
        
        # Perform Q-Q plot calculation
        osm, osr = stats.probplot(data, dist="norm")[0]  # Extract only the quantiles

        # Add scatter plot for each participant
        fig.add_trace(go.Scatter(
            x=osm, 
            y=osr, 
            mode='markers', 
            name=f'QQ Plot for {label}',
            marker=dict(size=6, opacity=0.7)
        ))

    # Add reference (y=x) line
    if fig.data:
        min_val = min(fig.data[0].x)
        max_val = max(fig.data[0].x)
    else:
        min_val, max_val = 0, 1

    fig.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val], 
        mode='lines', 
        name='Reference Line',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Ordered Values",
        template="plotly_white"
    )

    fig.show()
######################################################################################################################################################

def anderson_darling_test(df, columns, labels):
    """
    Performs the Anderson-Darling normality test for multiple participants.

    Parameters:
        df (DataFrame): The dataframe containing heart rate data.
        columns (list): List of column names for heart rate data.
        labels (list): List of corresponding participant labels.

    Returns:
        results (dict): A dictionary containing test results for each participant.
    """
    results = {}

    for col, label in zip(columns, labels):
        data = df[col].dropna()  # Drop NaN values

        if data.empty:
            print(f"Warning: No valid numeric data for {label}, skipping.")
            continue  # Skip if no valid data
        
        result = stats.anderson(data)  # Perform Anderson-Darling test

        # Store the results in a dictionary
        results[label] = {
            "Statistic": result.statistic,
            "Critical Values": result.critical_values,
            "Significance Levels": result.significance_level
        }

        # Print the results
        print(f"\n📌 Anderson-Darling Test for {label}:")
        print(f"   ➤ Statistic: {result.statistic:.4f}")
        print(f"   ➤ Critical Values: {result.critical_values}")
        print(f"   ➤ Significance Levels: {result.significance_level}%")
    
    return results
######################################################################################################################################################

def plot_average_heart_rate(df, columns, labels):
    """
    Generates an interactive bar chart for the average heart rate of each participant.

    Parameters:
        df (DataFrame): The dataframe containing heart rate data.
        columns (list): List of column names corresponding to heart rate measurements.
        labels (list): List of corresponding participant labels.

    Returns:
        fig (Plotly Figure): An interactive bar chart.
    """
    # Convert to numeric to avoid errors
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate means
    heart_rate_means = [df[col].mean() for col in columns]

    # Create DataFrame for plotting
    df_heart_rate = pd.DataFrame({'Person': labels, 'Average Heart-rate': heart_rate_means})

    # Create an interactive bar plot
    fig = px.bar(df_heart_rate, x='Person', y='Average Heart-rate', text='Average Heart-rate',
                 title='Average Heart Rate per Person', template="plotly_dark",
                 color='Average Heart-rate', color_continuous_scale='purples')

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    return fig
######################################################################################################################################################

def plot_heart_rate_distribution(df, columns, labels):
    """
    Generates histograms for the distribution of heart rate values for each participant.

    Parameters:
        df (DataFrame): The dataframe containing heart rate data.
        columns (list): List of column names corresponding to heart rate measurements.
        labels (list): List of corresponding participant labels.

    Returns:
        None (Displays histograms)
    """
    # Convert columns to numeric
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create a 2x2 subplot layout for 4 participants
    plt.figure(figsize=(12, 8))
    
    for i, (col, label) in enumerate(zip(columns, labels), 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True, bins=15, color='purple', edgecolor="black")
        plt.title(f'Distribution of Heart-rate - {label}')
        plt.xlabel('Heart-rate (bpm)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

######################################################################################################################################################
def plot_heart_rate_analysis(df):
    """
    Function to analyze heart rate changes over time using:
    1. Line Plot of heart rate across different dates.
    2. Box Plot to visualize heart rate distribution for each phase.
    3. Bar Chart showing the mean heart rate for each participant during different phases.

    Parameters:
        df (DataFrame): Dataframe containing heart rate data and Date column.
    """

    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Define participant heart rate columns
    heart_rate_columns = ['Heart-rate-Z', 'Heart-rate-R', 'Heart-rate-O', 'Heart-rate-J']
    participant_labels = ['Participant Ⅰ', 'Participant Ⅱ', 'Participant Ⅲ', 'Participant Ⅳ']
    
    # Convert heart rate columns to numeric
    for col in heart_rate_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extract date and heart rate data
    Date = df['Date']
    Heart_rate_Z = df['Heart-rate-Z']
    Heart_rate_R = df['Heart-rate-R']
    Heart_rate_O = df['Heart-rate-O']
    Heart_rate_J = df['Heart-rate-J']

    # Define phase boundaries and labels
    phases = [0, 10, 21, 41, 61]
    phase_labels = ['Half-Dose (10d)', 'Full-Dose (10d)', 'Half-Dose (20d)', 'Full-Dose (14d)']

    ### 📌 **1. Line Plot - Heart Rate Over Time**
    plt.figure(figsize=(15, 7))
    plt.plot(Date, Heart_rate_Z, label='Participant Ⅰ', marker='o')
    plt.plot(Date, Heart_rate_R, label='Participant Ⅱ', marker='o')
    plt.plot(Date, Heart_rate_O, label='Participant Ⅲ', marker='o')
    plt.plot(Date, Heart_rate_J, label='Participant Ⅳ', marker='o')

    # Add phase separators
    for p in phases[1:-1]:
        plt.axvline(x=Date.iloc[p], color='gray', linestyle='--', alpha=0.5)

    # Customize plot
    plt.title('Heart Rate Changes Over Time with Supplement Phases', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Heart Rate (bpm)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    ### 📌 **2. Box Plot - Heart Rate Distribution Across Phases**
    # Define data slices for each phase
    phase_data = {
        'Half-Dose (10d)': df.iloc[:10][heart_rate_columns],
        'Full-Dose (10d)': df.iloc[11:21][heart_rate_columns],
        'Half-Dose (20d)': df.iloc[22:42][heart_rate_columns],
        'Full-Dose (14d)': df.iloc[43:][heart_rate_columns]
    }

    # Create a boxplot
    plt.figure(figsize=(15, 7))
    for i, (phase, data) in enumerate(phase_data.items(), 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=data)
        plt.title(phase)
        plt.ylabel('Heart Rate (bpm)')
        plt.xticks(ticks=range(4), labels=participant_labels)

    plt.tight_layout()
    plt.show()

    ### 📌 **3. Bar Chart - Mean Heart Rate by Phase**
    means_df = pd.DataFrame({
        'Phase': phase_labels,
        'Ⅰ': [phase_data[phase]['Heart-rate-Z'].mean() for phase in phase_labels],
        'Ⅱ': [phase_data[phase]['Heart-rate-R'].mean() for phase in phase_labels],
        'Ⅲ': [phase_data[phase]['Heart-rate-O'].mean() for phase in phase_labels],
        'Ⅳ': [phase_data[phase]['Heart-rate-J'].mean() for phase in phase_labels]
    })

    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    r1 = np.arange(len(means_df['Phase']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    plt.bar(r1, means_df['Ⅰ'], width=bar_width, label='Ⅰ')
    plt.bar(r2, means_df['Ⅱ'], width=bar_width, label='Ⅱ')
    plt.bar(r3, means_df['Ⅲ'], width=bar_width, label='Ⅲ')
    plt.bar(r4, means_df['Ⅳ'], width=bar_width, label='Ⅳ')

    plt.xlabel('Phase')
    plt.ylabel('Mean Heart Rate (bpm)')
    plt.title('Mean Heart Rate by Phase and Participant')
    plt.xticks([r + bar_width*1.5 for r in range(len(means_df['Phase']))], means_df['Phase'], rotation=45)
    plt.legend()
    plt.show()
######################################################################################################################################################

def plot_heart_rate_boxplot(df):
    """
    Function to create a box plot for heart rate measurements across different participants.

    Parameters:
        df (DataFrame): Dataframe containing heart rate data.

    Output:
        A box plot visualizing the distribution of heart rate values for each participant.
    """

    # Define column names for Heart Rate
    heart_rate_columns = ['Heart-rate-Z', 'Heart-rate-R', 'Heart-rate-O', 'Heart-rate-J']
    roman_labels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']

    # Convert heart rate columns to numeric (to avoid errors)
    for col in heart_rate_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Set figure size
    plt.figure(figsize=(10, 6))

    # Create a box plot for heart rate columns
    sns.boxplot(data=df[heart_rate_columns])

    # Change x-axis labels to Roman numerals
    plt.xticks(ticks=range(4), labels=roman_labels)

    # Add title and labels
    plt.title("Box Plot of Heart Rate Variables")
    plt.ylabel("Heart Rate (bpm)")
    plt.xlabel("Participants")

    # Show the plot
    plt.show()

####################################################################### Inteactive ###############################################################################
# Function to update the box plot
def update_boxplot(selected_participant):
    plt.figure(figsize=(10, 5))
    
    if selected_participant == 'All Participants':
        sns.boxplot(data=df_long, x='Phase', y='Heart Rate', hue='Participant', palette='Set2')
        plt.title("Heart Rate Distribution Across Phases (All Participants)")
    else:
        subset = df_long[df_long['Participant'] == selected_participant]
        if subset.empty:
            print(f"No data available for {selected_participant}")
            return
        sns.boxplot(data=subset, x='Phase', y='Heart Rate', palette='Set2')
        plt.title(f"Heart Rate Distribution Across Phases for {selected_participant}")

    plt.xlabel("Time Phase")
    plt.ylabel("Heart Rate (bpm)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Participant")
    plt.show()

######################################################################################################################################################


def plot_heart_rate_regression(df_filled):
    """
    Generates an interactive scatter plot with regression lines for heart rate data.
    
    Parameters:
    df_filled (DataFrame): A Pandas DataFrame containing heart rate data.
    """
    
    # Ensure column names are clean
    df_filled.columns = df_filled.columns.str.strip()

    # Define column names for Heart-rate
    heart_rate_columns = ['Heart-rate-Z', 'Heart-rate-R', 'Heart-rate-O', 'Heart-rate-J']
    roman_labels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']

    # Check if 'Date' column exists for proper x-axis
    if 'Date' in df_filled.columns:
        df_filled['Date'] = pd.to_datetime(df_filled['Date'], errors='coerce')  # Convert to datetime format
        x_axis_values = df_filled['Date']
    else:
        x_axis_values = df_filled.index  # Fallback to index if no Date column

    # Create figure
    fig = go.Figure()

    # Creating dropdown options
    dropdown_options = []
    traces = []

    for i, (heart_col, roman) in enumerate(zip(heart_rate_columns, roman_labels)):
        heart_data = pd.to_numeric(df_filled[heart_col], errors='coerce').dropna()
        aligned_x = x_axis_values[df_filled[heart_col].notna()]  # Ensure correct x-values

        # Linear regression for Heart-rate
        if len(heart_data) > 1:
            heart_slope, heart_intercept, _, _, _ = stats.linregress(range(len(heart_data)), heart_data)
            reg_line = heart_slope * np.arange(len(heart_data)) + heart_intercept
        else:
            reg_line = np.full_like(heart_data, heart_data.mean()) if not heart_data.empty else []

        # Scatter plot for Heart-rate
        scatter_trace = go.Scatter(
            x=aligned_x, 
            y=heart_data, 
            mode='markers', 
            name=f'Heart-rate {roman}',
            marker=dict(size=6, opacity=0.7),
            visible=(i == 0)
        )
        traces.append(scatter_trace)

        # Regression line for Heart-rate
        reg_trace = go.Scatter(
            x=aligned_x, 
            y=reg_line,
            mode='lines',
            name=f'Regression {roman}',
            line=dict(dash='dash'),
            visible=(i == 0)
        )
        traces.append(reg_trace)

        dropdown_options.append(
            dict(label=roman,
                 method="update",
                 args=[{"visible": [j // 2 == i for j in range(2 * len(roman_labels))]},
                       {"title": f"Scatter Plot & Regression - {roman}"}])
        )

    # Add all traces to the figure
    fig.add_traces(traces)

    fig.update_layout(
        title="Scatter Plot with Regression for Heart-rate",
        xaxis_title="Time" if 'Date' in df_filled.columns else "Index",
        yaxis_title="Heart Rate (bpm)",
        updatemenus=[dict(buttons=dropdown_options, direction="down", showactive=True)],
        template="plotly_white"
    )

    fig.show()
######################################################################################################################################################
def plot_heart_rate_bar_chart(df_filled):
    """
    Generates an interactive bar chart to display mean heart rate for different users across phases.

    Parameters:
    df_filled (DataFrame): A Pandas DataFrame containing heart rate data.
    """
    # Ensure column names are clean
    df_filled.columns = df_filled.columns.str.strip()

    # Define heart rate measurement columns
    heart_rate_columns = {
        'I': 'Heart-rate-Z',
        'II': 'Heart-rate-R',
        'III': 'Heart-rate-O',
        'IV': 'Heart-rate-J'
    }

    # Create an output widget for rendering the plots
    output = widgets.Output()
#####################################################################################################################################################
    # Function to generate a bar chart for the selected user
    def analyze_user(selected_user):
        with output:
            output.clear_output(wait=True)  # Clear previous output

            heart_rate_col = heart_rate_columns[selected_user]

            # Filter and clean dataset
            user_data = df_filled[['Intake-Dose', heart_rate_col]].dropna()
            user_data = user_data[user_data['Intake-Dose'] != 'Break']
            user_data.rename(columns={heart_rate_col: 'Value', 'Intake-Dose': 'Phase'}, inplace=True)
            user_data['Phase'] = user_data['Phase'].str.strip()

            # Calculate mean value per phase
            mean_values = user_data.groupby('Phase')['Value'].mean()

            # Create a bar chart
            plt.figure(figsize=(8, 5))
            bars = plt.bar(mean_values.index, mean_values.values, color=['blue', 'green', 'orange', 'red'])

            # Add numerical values on top of bars
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.xlabel('Phase')
            plt.ylabel('Mean Heart Rate (bpm)')
            plt.title(f'Heart Rate - User {selected_user}')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    # Create a dropdown widget for user selection
    user_dropdown = widgets.Dropdown(
        options=list(heart_rate_columns.keys()),
        description='Select User:',
        style={'description_width': 'initial'}
    )

    # Attach observer to dropdown
    user_dropdown.observe(lambda change: analyze_user(change.new), names='value')

    # Display widgets
    display(user_dropdown, output)

    # Trigger initial plot display
    analyze_user(user_dropdown.value)



######################################################################################################################################################
    # Function to update the box plot
    def update_boxplot(selected_participant):
        with output:
            output.clear_output(wait=True)  # Clear previous output

            plt.figure(figsize=(10, 5))
            
            if selected_participant == 'All Participants':
                sns.boxplot(data=df_long, x='Phase', y='Heart Rate', hue='Participant', palette='Set2')
                plt.title("Heart Rate Distribution Across Phases (All Participants)")
            else:
                subset = df_long[df_long['Participant'] == selected_participant]
                if subset.empty:
                    print(f"No data available for {selected_participant}")
                    return
                sns.boxplot(data=subset, x='Phase', y='Heart Rate', palette='Set2')
                plt.title(f"Heart Rate Distribution Across Phases for {selected_participant}")

            plt.xlabel("Time Phase")
            plt.ylabel("Heart Rate (bpm)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(title="Participant")
            plt.show()

    # Create dropdown for participant selection
    dropdown = widgets.Dropdown(
        options=['All Participants'] + list(heart_rate_columns.keys()),
        description='Select Participant:',
        style={'description_width': 'initial'}
    )

    # Attach observer to dropdown
    dropdown.observe(lambda change: update_boxplot(change.new), names='value')

    # Display dropdown and output
    display(dropdown, output)

    # Trigger initial plot display
    update_boxplot(dropdown.value)


def interactive_statistics(df_filled):
    """
    Creates an interactive widget for statistical analysis of heart rate data.
    
    Parameters:
    df_filled (DataFrame): A Pandas DataFrame containing heart rate and phase data.
    """

    # Ensure column names are clean
    df_filled.columns = df_filled.columns.str.strip()

    # Define heart rate measurement columns
    heart_rate_columns = {
        'I': 'Heart-rate-Z',
        'II': 'Heart-rate-R',
        'III': 'Heart-rate-O',
        'IV': 'Heart-rate-J'
    }

    # Convert heart rate values to numeric
    for col in heart_rate_columns.values():
        df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')

    # Create output widget
    output = widgets.Output()


######################################################################################################################################################
def interactive_statistics(df_filled):
    """
    Creates an interactive widget for statistical analysis of heart rate data.
    
    Parameters:
    df_filled (DataFrame): A Pandas DataFrame containing heart rate and phase data.
    """

    # Ensure column names are clean
    df_filled.columns = df_filled.columns.str.strip()

    # Define heart rate measurement columns
    heart_rate_columns = {
        'I': 'Heart-rate-Z',
        'II': 'Heart-rate-R',
        'III': 'Heart-rate-O',
        'IV': 'Heart-rate-J'
    }

    # Convert heart rate values to numeric
    for col in heart_rate_columns.values():
        df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')

    # Create output widget
    output = widgets.Output()

    # Function to compute statistics
    def update_statistics(selected_participant):
        with output:
            output.clear_output(wait=True)  # Clear previous output
            
            if selected_participant == 'All Participants':
                selected_columns = list(heart_rate_columns.values())
            else:
                selected_columns = [heart_rate_columns[selected_participant]]

            if not selected_columns:
                print("No data available for selected participant.")
                return

            # Compute Summary Statistics
            summary_stats = df_filled[selected_columns].describe().T
            display(summary_stats)

            # Normality Test (Shapiro-Wilk)
            normality_results = {col: stats.shapiro(df_filled[col].dropna()).pvalue for col in selected_columns}
            normality_df = pd.DataFrame({
                "Participant": selected_columns,
                "Shapiro-W p-value": [normality_results[col] for col in selected_columns]
            })
            print("\n📊 **Normality Test (Shapiro-Wilk p-values)**")
            display(normality_df)

            # Homogeneity Test (Levene's Test)
            if len(selected_columns) > 1:
                levene_test = stats.levene(*(df_filled[col].dropna() for col in selected_columns))
                homogeneity_p_value = levene_test.pvalue
            else:
                homogeneity_p_value = "N/A (Only one participant selected)"

            # ANOVA Test
            if len(selected_columns) > 1:
                anova_test = stats.f_oneway(*(df_filled[col].dropna() for col in selected_columns))
                anova_p_value = anova_test.pvalue
            else:
                anova_p_value = "N/A (Only one participant selected)"

            print(f"\n📌 **Levene's Test for Homogeneity of Variance p-value:** {homogeneity_p_value}")
            print(f"📌 **ANOVA Test for Mean Comparison p-value:** {anova_p_value}")

    # Create dropdown for participant selection
    dropdown = widgets.Dropdown(
        options=['All Participants'] + list(heart_rate_columns.keys()),
        description='Select Participant:',
        style={'description_width': 'initial'}
    )

    # Attach observer to dropdown
    dropdown.observe(lambda change: update_statistics(change.new), names='value')

    # Display dropdown and output
    display(dropdown, output)

    # Trigger initial display
    update_statistics(dropdown.value)

######################################################################################################################################################


# ✅ Function to Prepare Data
def prepare_data(df):
    # Ensure column names are clean
    df.columns = df.columns.str.strip()
    
    # Define heart rate measurement columns
    heart_rate_columns = ['Heart-rate-Z', 'Heart-rate-R', 'Heart-rate-O', 'Heart-rate-J']
    
    # Convert heart rate values to numeric
    for col in heart_rate_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure 'Date' column is in datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Define phases
    num_rows = len(df)
    phase_labels = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
    df['Phase'] = np.tile(phase_labels, num_rows // len(phase_labels) + 1)[:num_rows]

    # Convert DataFrame into long format
    df_long = pd.melt(df, id_vars=['Date', 'Phase'], value_vars=heart_rate_columns, var_name='Participant', value_name='Heart Rate')

    # Map participant names
    participant_mapping = {
        'Heart-rate-Z': 'I',
        'Heart-rate-R': 'II',
        'Heart-rate-O': 'III',
        'Heart-rate-J': 'IV'
    }
    df_long['Participant'] = df_long['Participant'].map(participant_mapping)
    
    return df_long
# ✅ Function to Create Interactive Boxplot
def create_interactive_boxplot(df_long):
    # Create dropdown for participant selection
    participants = ['All Participants'] + list(df_long['Participant'].unique())

    dropdown = widgets.Dropdown(
        options=participants,
        description='Select Participant:',
        style={'description_width': 'initial'}
    )

    # Function to update the box plot
    def update_boxplot(selected_participant):
        plt.figure(figsize=(10, 5))
        
        if selected_participant == 'All Participants':
            sns.boxplot(data=df_long, x='Phase', y='Heart Rate', hue='Participant', palette='Set2')
            plt.title("Heart Rate Distribution Across Phases (All Participants)")
        else:
            subset = df_long[df_long['Participant'] == selected_participant]
            if subset.empty:
                print(f"No data available for {selected_participant}")
                return
            sns.boxplot(data=subset, x='Phase', y='Heart Rate', palette='Set2')
            plt.title(f"Heart Rate Distribution Across Phases for {selected_participant}")

        plt.xlabel("Time Phase")
        plt.ylabel("Heart Rate (bpm)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Participant")
        plt.show()

    # Display dropdown and connect it to function
    interactive_plot = widgets.interactive(update_boxplot, selected_participant=dropdown)
    display(dropdown, interactive_plot)

###################################################################################################################################################
# ✅ Function to Generate QQ Plots for Blood Pressure
def generate_qq_plots(df):
    # Define column names for each metric
    systolic_bp_columns = [
        'Blood-Pressure(systolic)(mmHg)-Z', 
        'Blood-Pressure(systolic)(mmHg)-R', 
        'Blood-Pressure(systolic)(mmHg)-O', 
        'Blood-Pressure(systolic)(mmHg)-J'
    ]

    diastolic_bp_columns = [
        'Blood-Pressure(diastolic)(mmHg)-Z', 
        'Blood-Pressure(diastolic)(mmHg)-R', 
        'Blood-Pressure(diastolic)(mmHg)-O', 
        'Blood-Pressure(diastolic)(mmHg)-J'
    ]

    participant_labels = ['User I', 'User II', 'User III', 'User IV']

    # Ensure Blood Pressure values are numeric
    for col in systolic_bp_columns + diastolic_bp_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ Generate QQ Plots for Systolic Blood Pressure
    plt.figure(figsize=(12, 8))
    for i, (col, label) in enumerate(zip(systolic_bp_columns, participant_labels), 1):
        plt.subplot(2, 2, i)
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        plt.title(f'QQ Plot for Systolic BP - {label}')
    
    plt.tight_layout()
    plt.show()

    # ✅ Generate QQ Plots for Diastolic Blood Pressure
    plt.figure(figsize=(12, 8))
    for i, (col, label) in enumerate(zip(diastolic_bp_columns, participant_labels), 1):
        plt.subplot(2, 2, i)
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        plt.title(f'QQ Plot for Diastolic BP - {label}')
    
    plt.tight_layout()
    plt.show()

###################################################################################################################################################
# ✅ Function to Calculate & Plot Systolic and Diastolic Blood Pressure
def plot_blood_pressure(df):
    # Ensure column names are clean
    df.columns = df.columns.str.strip()

    # Define column names for each metric
    systolic_bp_columns = [
        'Blood-Pressure(systolic)(mmHg)-Z', 
        'Blood-Pressure(systolic)(mmHg)-R', 
        'Blood-Pressure(systolic)(mmHg)-O', 
        'Blood-Pressure(systolic)(mmHg)-J'
    ]

    diastolic_bp_columns = [
        'Blood-Pressure(diastolic)(mmHg)-Z', 
        'Blood-Pressure(diastolic)(mmHg)-R', 
        'Blood-Pressure(diastolic)(mmHg)-O', 
        'Blood-Pressure(diastolic)(mmHg)-J'
    ]

    participant_labels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']

    # ✅ Check if columns exist in the dataset
    missing_cols = [col for col in systolic_bp_columns + diastolic_bp_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in dataset: {missing_cols}")
        return  # Stop execution if columns are missing

    # ✅ Convert columns to numeric values
    for col in systolic_bp_columns + diastolic_bp_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ Calculate means for each user and handle NaN values
    systolic_means = [df[col].mean() for col in systolic_bp_columns]
    diastolic_means = [df[col].mean() for col in diastolic_bp_columns]

    # ✅ Replace NaN values with 0 for visualization
    systolic_means = [0 if pd.isna(val) else val for val in systolic_means]
    diastolic_means = [0 if pd.isna(val) else val for val in diastolic_means]

    # ✅ Create DataFrames for visualization
    df_systolic = pd.DataFrame({'Person': participant_labels, 'Systolic BP (mmHg)': systolic_means})
    df_diastolic = pd.DataFrame({'Person': participant_labels, 'Diastolic BP (mmHg)': diastolic_means})

    # ✅ Interactive bar plot for Systolic Blood Pressure
    fig_systolic = px.bar(
        df_systolic, x='Person', y='Systolic BP (mmHg)', text='Systolic BP (mmHg)',
        title='Average Systolic Blood Pressure (mmHg) per Person', template="plotly_dark",
        color='Systolic BP (mmHg)', color_continuous_scale='reds'
    )
    fig_systolic.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_systolic.show()

    # ✅ Interactive bar plot for Diastolic Blood Pressure
    fig_diastolic = px.bar(
        df_diastolic, x='Person', y='Diastolic BP (mmHg)', text='Diastolic BP (mmHg)',
        title='Average Diastolic Blood Pressure (mmHg) per Person', template="plotly_dark",
        color='Diastolic BP (mmHg)', color_continuous_scale='blues'
    )
    fig_diastolic.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_diastolic.show()

###################################################################################################################################################

# ✅ Function to Plot Blood Pressure Distributions
def plot_blood_pressure_distribution(df):
    # Ensure column names are clean
    df.columns = df.columns.str.strip()

    # Define column names
    systolic_columns = [
        'Blood-Pressure(systolic)(mmHg)-Z', 
        'Blood-Pressure(systolic)(mmHg)-R', 
        'Blood-Pressure(systolic)(mmHg)-O', 
        'Blood-Pressure(systolic)(mmHg)-J'
    ]
    diastolic_columns = [
        'Blood-Pressure(diastolic)(mmHg)-Z', 
        'Blood-Pressure(diastolic)(mmHg)-R', 
        'Blood-Pressure(diastolic)(mmHg)-O', 
        'Blood-Pressure(diastolic)(mmHg)-J'
    ]
    roman_labels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']

    # ✅ Convert columns to numeric
    for col in systolic_columns + diastolic_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 🔵 Plot Systolic Blood Pressure Distribution
    plt.figure(figsize=(12, 8))
    for i, (col, roman) in enumerate(zip(systolic_columns, roman_labels), 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True, bins=15, color='blue')
        plt.title(f'Distribution of Systolic Blood Pressure - {roman}')
        plt.xlabel('Systolic BP (mmHg)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # 🔴 Plot Diastolic Blood Pressure Distribution
    plt.figure(figsize=(12, 8))
    for i, (col, roman) in enumerate(zip(diastolic_columns, roman_labels), 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True, bins=15, color='red')
        plt.title(f'Distribution of Diastolic Blood Pressure - {roman}')
        plt.xlabel('Diastolic BP (mmHg)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# ✅ Function to plot box plots for Blood Pressure
def plot_blood_pressure_boxplots(df):
    # Ensure column names are clean
    df.columns = df.columns.str.strip()

    # Define column names
    systolic_columns = [
        'Blood-Pressure(systolic)(mmHg)-Z', 
        'Blood-Pressure(systolic)(mmHg)-R', 
        'Blood-Pressure(systolic)(mmHg)-O', 
        'Blood-Pressure(systolic)(mmHg)-J'
    ]
    diastolic_columns = [
        'Blood-Pressure(diastolic)(mmHg)-Z', 
        'Blood-Pressure(diastolic)(mmHg)-R', 
        'Blood-Pressure(diastolic)(mmHg)-O', 
        'Blood-Pressure(diastolic)(mmHg)-J'
    ]
    roman_labels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']

    # Convert to numeric
    for col in systolic_columns + diastolic_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 🔵 Systolic Blood Pressure Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[systolic_columns])
    plt.xticks(ticks=range(4), labels=roman_labels)
    plt.title("Box Plot of Blood Pressure (Systolic)")
    plt.ylabel("Blood Pressure (mmHg)")
    plt.show()

    # 🔴 Diastolic Blood Pressure Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[diastolic_columns])
    plt.xticks(ticks=range(4), labels=roman_labels)
    plt.title("Box Plot of Blood Pressure (Diastolic)")
    plt.ylabel("Blood Pressure (mmHg)")
    plt.show()