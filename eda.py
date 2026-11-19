"""
Aiden Hoang
CSE 163 Final Project, Part 2: Exploratory Data Analysis

Runs the EDA for the Seattle bike share (Pronto) and weather project.
It loads trip.csv, weather.csv, and station.csv from the data/ folder,
cleans them, builds the joined datasets used by each research question,
prints the size, missing data, and summary statistics for each one,
checks the assumptions for the statistical tests planned for the Result
Validity challenge goal, and saves every plot as a .png file.

Joined datasets:
    1. daily trips + weather (joined on date), for the temperature question
    2. trips on their own, for the weekday vs. weekend ride length question
    3. trips + stations (joined on starting station id), for the
       neighborhood question
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


DATA_DIR = 'data/'

# The first part of every station id is a neighborhood code, for example
# 'CH-07' is a Capitol Hill station. This maps each code to a readable name.
NEIGHBORHOODS = {
    'BT': 'Belltown',
    'CBD': 'Downtown',
    'CD': 'Central District',
    'CH': 'Capitol Hill',
    'DPD': 'Other (scattered)',
    'EL': 'Eastlake',
    'FH': 'First Hill',
    'ID': 'International District',
    'PS': 'Pioneer Square',
    'SLU': 'South Lake Union',
    'UD': 'University District',
    'UW': 'UW Campus',
    'WF': 'Waterfront',
}


def label_day_type(dates):
    """
    Takes a column of date strings like '10/13/2014' and returns a list
    with 'Weekday' or 'Weekend' for each one. This is the only place the
    code uses pd.to_datetime, because a date string alone does not say
    which day of the week it is (Monday is 0, Saturday is 5, Sunday is 6).
    """
    day_numbers = pd.to_datetime(dates, format='%m/%d/%Y').dt.dayofweek
    labels = []
    for day in day_numbers:
        if day >= 5:
            labels.append('Weekend')
        else:
            labels.append('Weekday')
    return labels


def month_label(date):
    """
    Takes a date string like '6/14/2015' and returns its month as a string
    like '2015-06', so that sorting the labels also sorts them by time.
    """
    parts = date.split('/')
    month = parts[0]
    if len(month) == 1:
        month = '0' + month
    return parts[2] + '-' + month


def load_trips(path):
    """
    Reads the trip file and returns a DataFrame with extra columns:
    'date' (the date part of the start time, like '10/13/2014'),
    'minutes' (trip length in minutes), and 'day_type' ('Weekday' or
    'Weekend'). Lines that do not have the right number of fields are
    skipped.
    """
    trips = pd.read_csv(path, on_bad_lines='skip')
    trips['date'] = [start.split(' ')[0] for start in trips['starttime']]
    trips['minutes'] = trips['tripduration'] / 60
    trips['day_type'] = label_day_type(trips['date'])
    return trips


def load_weather(path):
    """
    Reads the weather file and adds 'month' (like '2015-06') and
    'day_type' columns. It also cleans two things: a '-' in the gust
    column is treated as missing, and a blank Events value means nothing
    happened that day, so it becomes 'None'. The one missing mean
    temperature is filled with the average of that day's max and min.
    Returns the cleaned DataFrame.
    """
    weather = pd.read_csv(path, na_values='-')
    weather['Events'] = weather['Events'].fillna('None')
    weather['month'] = [month_label(date) for date in weather['Date']]
    weather['day_type'] = label_day_type(weather['Date'])
    midpoint = (weather['Max_Temperature_F'] +
                weather['Min_TemperatureF']) / 2
    missing = weather['Mean_Temperature_F'].isna()
    weather.loc[missing, 'Mean_Temperature_F'] = midpoint[missing]
    return weather


def load_stations(path):
    """
    Reads the station file and adds a 'neighborhood' column based on the
    code at the front of each station id. Returns the DataFrame.
    """
    stations = pd.read_csv(path)
    neighborhoods = []
    for station_id in stations['station_id']:
        code = station_id.split('-')[0]
        neighborhoods.append(NEIGHBORHOODS[code])
    stations['neighborhood'] = neighborhoods
    return stations


def daily_trip_weather(trips, weather):
    """
    Counts trips per day and joins the counts with that day's weather.
    Returns one row per day with a 'trip_count' column added to all the
    weather columns. Days in the weather file with no trips get 0.
    """
    counts = trips.groupby('date')['trip_id'].count()
    rows = []
    for date in counts.index:
        rows.append({'Date': date, 'trip_count': counts[date]})
    counts_df = pd.DataFrame(rows)
    daily = weather.merge(counts_df, left_on='Date', right_on='Date',
                          how='left')
    daily['trip_count'] = daily['trip_count'].fillna(0)
    return daily


def trips_with_stations(trips, stations):
    """
    Joins each trip with the station it started at. Trips that start at
    a place that is not in the station file (the Pronto repair shop and
    an operations id) are dropped by the inner join. Returns the result.
    """
    keep = stations[['station_id', 'name', 'neighborhood', 'lat', 'long',
                     'current_dockcount']]
    return trips.merge(keep, left_on='from_station_id',
                       right_on='station_id', how='inner')


def missing_report(df):
    """
    Returns a DataFrame with the number and percent of missing values for
    every column that has at least one missing value.
    """
    counts = df.isna().sum()
    rows = []
    for column in counts.index:
        if counts[column] > 0:
            rows.append({'column': column, 'missing': counts[column],
                         'percent': counts[column] / len(df) * 100})
    return pd.DataFrame(rows)


def seven_number_summary(values):
    """
    Returns a dictionary with the mean, standard deviation, minimum, first
    quartile, median, third quartile, and maximum of the given column,
    taken from the column's describe() results.
    """
    described = values.describe()
    return {
        'mean': described['mean'],
        'std': described['std'],
        'min': described['min'],
        'Q1': described['25%'],
        'median': described['50%'],
        'Q3': described['75%'],
        'max': described['max'],
    }


def print_summary(name, values):
    """Prints the seven-number summary of a column, rounded to 2 places."""
    print(name)
    summary = seven_number_summary(values)
    for key in summary:
        print('  ' + key + ':', round(summary[key], 2))


def top_neighborhoods(joined):
    """
    Returns a Series of trip counts per starting neighborhood, sorted from
    most to fewest trips.
    """
    counts = joined.groupby('neighborhood')['trip_id'].count()
    return counts.sort_values(ascending=False)


def trips_per_station(joined):
    """
    Returns a Series of trip counts per starting station name, sorted from
    most to fewest trips.
    """
    counts = joined.groupby('name')['trip_id'].count()
    return counts.sort_values(ascending=False)


def plot_temp_vs_trips(daily, path):
    """Scatter plot of daily mean temperature vs. trips that day."""
    fig, ax = plt.subplots(1, figsize=(6.4, 4))
    sns.scatterplot(data=daily, x='Mean_Temperature_F', y='trip_count',
                    hue='day_type', ax=ax)
    ax.set_title('Daily mean temperature vs. number of bike trips')
    ax.set_xlabel('Mean temperature (°F)')
    ax.set_ylabel('Trips that day')
    fig.savefig(path)


def plot_trips_over_time(daily, path):
    """
    Two stacked line plots: average trips per day in each month on top,
    average temperature in each month on the bottom.
    """
    trips = daily.groupby('month')['trip_count'].mean()
    temps = daily.groupby('month')['Mean_Temperature_F'].mean()
    fig, [ax1, ax2] = plt.subplots(2, figsize=(10, 6))
    sns.lineplot(x=list(trips.index), y=list(trips), marker='o', ax=ax1)
    ax1.set_title('Monthly average trips per day')
    ax1.set_ylabel('Avg trips per day')
    ax1.tick_params(axis='x', labelrotation=90)
    sns.lineplot(x=list(temps.index), y=list(temps), marker='o',
                 color='orange', ax=ax2)
    ax2.set_title('Monthly average temperature')
    ax2.set_ylabel('Avg temp (°F)')
    ax2.set_xlabel('Month')
    ax2.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    fig.savefig(path)


def plot_duration_hist(trips, path):
    """Histogram of trip length (up to 90 minutes) by day type."""
    short = trips[trips['minutes'] <= 90]
    fig, ax = plt.subplots(1, figsize=(6.4, 4))
    sns.histplot(data=short, x='minutes', hue='day_type', stat='density',
                 common_norm=False, binwidth=2, ax=ax)
    ax.set_title('Trip length, weekday vs. weekend')
    ax.set_xlabel('Trip length (minutes, up to 90 shown)')
    ax.set_ylabel('Share of trips (density)')
    fig.savefig(path)


def plot_duration_box(trips, path):
    """Box plot of trip length by day type and user type."""
    fig, ax = plt.subplots(1, figsize=(6.4, 4))
    sns.boxplot(data=trips, x='usertype', y='minutes', hue='day_type',
                showfliers=False, ax=ax)
    ax.set_title('Trip length by day and rider type (outliers hidden)')
    ax.set_xlabel('Rider type')
    ax.set_ylabel('Trip length (minutes)')
    fig.savefig(path)


def plot_neighborhoods(counts, path):
    """Bar chart of trips starting in each neighborhood."""
    fig, ax = plt.subplots(1, figsize=(8, 5))
    sns.barplot(x=list(counts), y=list(counts.index), ax=ax)
    ax.set_title('Trips by starting neighborhood')
    ax.set_xlabel('Number of trips started')
    fig.tight_layout()
    fig.savefig(path)


def plot_top_stations(station_counts, path):
    """Bar chart of the 10 stations where the most trips start."""
    top = station_counts.head(10)
    fig, ax = plt.subplots(1, figsize=(9, 5))
    sns.barplot(x=list(top), y=list(top.index), ax=ax)
    ax.set_title('Top 10 starting stations')
    ax.set_xlabel('Number of trips started')
    fig.tight_layout()
    fig.savefig(path)


def plot_rq1_assumptions(daily, path):
    """
    Histograms of daily trip counts and daily mean temperature, used to
    check the normality assumption for the Pearson correlation test.
    """
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(data=daily, x='trip_count', bins=30, ax=ax1)
    ax1.set_title('Trips per day')
    ax1.set_xlabel('Trips')
    sns.histplot(data=daily, x='Mean_Temperature_F', bins=30,
                 color='orange', ax=ax2)
    ax2.set_title('Mean temperature')
    ax2.set_xlabel('°F')
    fig.savefig(path)


def check_rq1_assumptions(daily):
    """
    Checks the assumptions of a Pearson correlation between daily mean
    temperature and daily trip count. Returns a dictionary with the
    Shapiro-Wilk p-values, the correlation between each day's trip count
    and the day before's (to check independence), and the Pearson and
    Spearman correlations (as a preview, not the final test).
    """
    temp = daily['Mean_Temperature_F']
    count = daily['trip_count']
    count_list = list(count)
    today = pd.Series(count_list[1:])
    yesterday = pd.Series(count_list[:-1])
    return {
        'shapiro_p_trips': stats.shapiro(count).pvalue,
        'shapiro_p_temp': stats.shapiro(temp).pvalue,
        'day_to_day_corr': today.corr(yesterday),
        'pearson_r': temp.corr(count),
        'spearman_rho': temp.corr(count, method='spearman'),
    }


def check_rq2_assumptions(trips):
    """
    Checks the assumptions of a two-sample t-test on trip length between
    weekdays and weekends. Returns a dictionary with group sizes, the
    skewness of each group, and the Levene test p-value for equal
    variances.
    """
    weekday = trips[trips['day_type'] == 'Weekday']['minutes']
    weekend = trips[trips['day_type'] == 'Weekend']['minutes']
    return {
        'n_weekday': len(weekday),
        'n_weekend': len(weekend),
        'skew_weekday': weekday.skew(),
        'skew_weekend': weekend.skew(),
        'levene_p': stats.levene(weekday, weekend).pvalue,
    }


def print_results(name, results):
    """
    Prints each entry of a results dictionary, rounded to 4 places.
    Very small values (like tiny p-values) are printed without rounding
    so they do not show up as 0.
    """
    print(name)
    for key in results:
        value = results[key]
        if value != 0 and abs(value) < 0.001:
            print('  ' + key + ':', value)
        else:
            print('  ' + key + ':', round(value, 4))


def print_header(text):
    """Prints a section header so the output is easy to read."""
    print()
    print('=' * 70)
    print(text)
    print('=' * 70)


def describe_dataset(name, df):
    """Prints the size and missing-data report for a dataset."""
    print_header(name)
    print('Rows, columns:', df.shape)
    print('Columns:', list(df.columns))
    report = missing_report(df)
    if len(report) == 0:
        print('No missing values')
    else:
        print('Missing values:')
        print(report)


def main():
    with open(DATA_DIR + 'trip.csv') as f:
        raw_lines = len(f.readlines()) - 1
    trips = load_trips(DATA_DIR + 'trip.csv')
    weather = load_weather(DATA_DIR + 'weather.csv')
    stations = load_stations(DATA_DIR + 'station.csv')
    print('Data lines in trip.csv:', raw_lines)
    print('Trips loaded:', len(trips), '| skipped:', raw_lines - len(trips))
    raw_weather = pd.read_csv(DATA_DIR + 'weather.csv', na_values='-')
    print('Raw weather missing values:')
    print(missing_report(raw_weather))

    # Dataset 1: daily trips joined with weather
    daily = daily_trip_weather(trips, weather)
    describe_dataset('DATASET 1: daily trips + weather', daily)
    print('Days with zero trips:', len(daily[daily['trip_count'] == 0]))
    print()
    print_summary('trip_count', daily['trip_count'])
    print_summary('Mean_Temperature_F', daily['Mean_Temperature_F'])
    print_summary('Precipitation_In', daily['Precipitation_In'])
    print('\nday_type counts (days):')
    print(daily.groupby('day_type')['Date'].count())
    print()
    print_results('RQ1 assumption checks:', check_rq1_assumptions(daily))
    plot_temp_vs_trips(daily, 'temp_vs_trips.png')
    plot_trips_over_time(daily, 'trips_over_time.png')
    plot_rq1_assumptions(daily, 'rq1_hist.png')

    # Dataset 2: trips on their own
    describe_dataset('DATASET 2: trips', trips)
    print('\nSummary of minutes by day_type:')
    for day_type in ['Weekday', 'Weekend']:
        minutes = trips[trips['day_type'] == day_type]['minutes']
        print_summary(day_type, minutes)
    print('\nday_type counts (trips):')
    print(trips.groupby('day_type')['trip_id'].count())
    print('\nusertype counts:')
    print(trips.groupby('usertype')['trip_id'].count())
    print('\nMedian minutes by usertype and day_type:')
    print(trips.groupby(['usertype', 'day_type'])['minutes'].median())
    print('\nPercent of trips by short-term pass holders, by day_type:')
    for day_type in ['Weekday', 'Weekend']:
        day_trips = trips[trips['day_type'] == day_type]
        short_term = day_trips[day_trips['usertype'] ==
                               'Short-Term Pass Holder']
        print(' ', day_type, round(len(short_term) / len(day_trips) * 100,
                                   1))
    print('Trips over 3 hours:', len(trips[trips['minutes'] > 180]))
    print()
    print_results('RQ2 assumption checks:', check_rq2_assumptions(trips))
    plot_duration_hist(trips, 'duration_hist.png')
    plot_duration_box(trips, 'duration_box.png')

    # Dataset 3: trips joined with stations
    joined = trips_with_stations(trips, stations)
    describe_dataset('DATASET 3: trips + stations', joined)
    print('Trips dropped by join (not a real station):',
          len(trips) - len(joined))
    print('\nTrips per neighborhood:')
    hoods = top_neighborhoods(joined)
    print(hoods)
    station_counts = trips_per_station(joined)
    print('\nNumber of starting stations:', len(station_counts))
    print('\nTop 10 starting stations:')
    print(station_counts.head(10))
    print()
    print_summary('Trips per station', station_counts)
    plot_neighborhoods(hoods, 'neighborhoods.png')
    plot_top_stations(station_counts, 'top_stations.png')


if __name__ == '__main__':
    main()
