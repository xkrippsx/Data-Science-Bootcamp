import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD"
data = pd.read_csv(url)

# Convert the 'hour_beginning' column to datetime type
data['hour_beginning'] = pd.to_datetime(data['hour_beginning'])

# Filter data for weekdays (Monday to Friday)
weekdays_data = data[data['hour_beginning'].dt.dayofweek < 5]

# Group by day of the week and sum pedestrian counts
daily_counts = weekdays_data.groupby(weekdays_data['hour_beginning'].dt.dayofweek)['Pedestrians'].sum()

# Define labels for days of the week
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(day_labels, daily_counts)
plt.title('Pedestrian Counts on Weekdays')
plt.xlabel('Day of the Week')
plt.ylabel('Pedestrian Count')
plt.show()

# Task 2: Analyze pedestrian counts on the Brooklyn Bridge for the year 2019 by weather conditions

# Filter data for the year 2019 and Brooklyn Bridge location
brooklyn_bridge_data = data[(data['hour_beginning'].dt.year == 2019)& (data['location'] == 'Brooklyn Bridge')]

# Convert 'Pedestrians' column to numeric, coercing errors to NaN for non-numeric values
brooklyn_bridge_data['Pedestrians'] = pd.to_numeric(brooklyn_bridge_data['Pedestrians'], errors='coerce')

# Drop rows with NaN values in 'Pedestrians' column
brooklyn_bridge_data = brooklyn_bridge_data.dropna(subset=['Pedestrians'])

# Group by weather summary and sum pedestrian counts
weather_pedestrian_counts = brooklyn_bridge_data.groupby('weather_summary')['Pedestrians'].sum().reset_index()
# Sort by pedestrian counts
sorted_weather_counts = weather_pedestrian_counts.sort_values('Pedestrians')

# Create correlation matrix
correlation_matrix = brooklyn_bridge_data.corr()

# Plot correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Weather and Pedestrian Counts on Brooklyn Bridge (2019)')
plt.show()

# Task 3: Implement a custom function to categorize time of day and analyze pedestrian activity patterns

def categorize_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 20:
        return 'Evening'
    else:
        return 'Night'

# Create a new column for time of day category
data['Time of Day'] = data['hour_beginning'].dt.hour.apply(categorize_time_of_day)

# Analyze pedestrian activity patterns throughout the day
time_of_day_counts = data.groupby('Time of Day')['Pedestrians'].sum()

# Plotting pedestrian activity patterns
plt.figure(figsize=(8, 6))
time_of_day_counts.plot(kind='bar', color='skyblue')
plt.title('Pedestrian Activity Patterns Throughout the Day')
plt.xlabel('Time of Day')
plt.ylabel('Pedestrian Count')
plt.xticks(rotation=45)
plt.show()
