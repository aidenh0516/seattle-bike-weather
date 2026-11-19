"""
Aiden Hoang
CSE 163 Final Project, Part 2: Exploratory Data Analysis

Runs the EDA for the Seattle bike share (Pronto) and weather project.
It loads trip.csv, weather.csv, and station.csv from the data/ folder,
cleans them, builds the joined datasets used by each research question,
prints the size / missing data / summary statistics for each one, checks
the assumptions for the statistical tests planned for the Result Validity
challenge goal, and saves every plot to the figures/ folder.

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


DATA_DIR = 'data'
FIG_DIR = 'figures'

WEEKDAY_COLOR = '#2a78d6'
WEEKEND_COLOR = '#eb6834'
DAY_TYPE_COLORS = {'Weekday': WEEKDAY_COLOR, 'Weekend': WEEKEND_COLOR}

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
    Takes a Series of date strings like '10/13/2014' and returns a Series
    of 'Weekday' or 'Weekend' for each one. This is the only place the
    code uses datetime, because a date string alone does not say which
    day of the week it is.
    """
    day_number = pd.to_datetime(dates, format='%m/%d/%Y').dt.dayofweek
    labels = pd.Series('Weekday', index=dates.index)
    labels[day_number >= 5] = 'Weekend'
    return labels


def load_trips(path):
    """
    Reads the trip file and returns a DataFrame with extra columns:
    'date' (the date part of the start time, like '10/13/2014'),
    'minutes' (trip length in minutes), and 'day_type' ('Weekday' or
    'Weekend'). Lines that do not have the right number of fields are
    skipped.
    """
    trips = pd.read_csv(path, on_bad_lines='skip')
    trips['date'] = trips['starttime'].str.split(' ').str[0]
    trips['minutes'] = trips['tripduration'] / 60
    trips['day_type'] = label_day_type(trips['date'])
    return trips


def load_weather(path):
    """
    Reads the weather file and adds 'month' (like '2015-06') and
    'day_type' columns. It also cleans two things: a '-'
    in the gust column is treated as missing, and a blank Events value
    means nothing happened that day, so it becomes 'None'. The one missing
    mean temperature is filled with the average of that day's max and min.
    Returns the cleaned DataFrame.
    """
    weather = pd.read_csv(path, na_values='-')
    weather['Events'] = weather['Events'].fillna('None')
    parts = weather['Date'].str.split('/')
    weather['month'] = parts.str[2] + '-' + parts.str[0].str.zfill(2)
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
    codes = stations['station_id'].str.split('-').str[0]
    stations['neighborhood'] = codes.map(NEIGHBORHOODS)
    return stations


def daily_trip_weather(trips, weather):
    """
    Counts trips per day and joins the counts with that day's weather.
    Returns one row per day with a 'trip_count' column added to all the
    weather columns. Days in the weather file with no trips get 0.
    """
    counts = trips.groupby('date').size().reset_index(name='trip_count')
    daily = weather.merge(counts, left_on='Date', right_on='date',
                          how='left')
    daily = daily.drop(columns='date')
    daily['trip_count'] = daily['trip_count'].fillna(0).astype(int)
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
    report = pd.DataFrame({'missing': counts,
                           'percent': counts / len(df) * 100})
    return report[report['missing'] > 0]


def seven_number_summary(series):
    """
    Returns a Series with the mean, standard deviation, minimum, first
    quartile, median, third quartile, and maximum of the given values.
    """
    return pd.Series({
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'Q1': series.quantile(0.25),
        'median': series.median(),
        'Q3': series.quantile(0.75),
        'max': series.max(),
    })


def top_neighborhoods(joined):
    """
    Returns a Series of trip counts per starting neighborhood, sorted from
    most to fewest trips.
    """
    counts = joined.groupby('neighborhood').size()
    return counts.sort_values(ascending=False)


def trips_per_station(joined):
    """
    Returns a DataFrame with one row per starting station: its name,
    neighborhood, and number of trips, sorted from most to fewest trips.
    """
    counts = joined.groupby(['name', 'neighborhood']).size()
    counts = counts.reset_index(name='trips')
    return counts.sort_values('trips', ascending=False)


def style_axes(ax):
    """Makes the grid and spines quiet so the data stands out."""
    ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)


def plot_temp_vs_trips(daily, path):
    """Scatter plot of daily mean temperature vs. trips that day."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    sns.scatterplot(data=daily, x='Mean_Temperature_F', y='trip_count',
                    hue='day_type', palette=DAY_TYPE_COLORS, s=22,
                    alpha=0.75, edgecolor='white', linewidth=0.4, ax=ax)
    ax.set_title('Daily mean temperature vs. number of bike trips')
    ax.set_xlabel('Mean temperature (°F)')
    ax.set_ylabel('Trips that day')
    ax.legend(title='Day type', frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_trips_over_time(daily, path):
    """
    Two stacked line plots sharing the x-axis: average trips per day in
    each month on top, average temperature in each month on the bottom.
    """
    monthly = daily.groupby('month')[['trip_count',
                                      'Mean_Temperature_F']].mean()
    positions = list(range(len(monthly)))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.6, 3.9), sharex=True)
    top.plot(positions, monthly['trip_count'], color=WEEKDAY_COLOR,
             linewidth=2, marker='o', markersize=3)
    top.set_ylabel('Avg trips per day')
    top.set_title('Monthly averages of daily trips and temperature')
    bottom.plot(positions, monthly['Mean_Temperature_F'],
                color=WEEKEND_COLOR, linewidth=2, marker='o', markersize=3)
    bottom.set_ylabel('Avg temp (°F)')
    bottom.set_xlabel('Month')
    bottom.set_xticks(positions[::4])
    bottom.set_xticklabels(list(monthly.index[::4]))
    style_axes(top)
    style_axes(bottom)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_duration_hist(trips, path):
    """Histogram of trip length (up to 90 minutes) by day type."""
    short = trips[trips['minutes'] <= 90]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    sns.histplot(data=short, x='minutes', hue='day_type',
                 palette=DAY_TYPE_COLORS, stat='density',
                 common_norm=False, binwidth=2, element='step', fill=False,
                 linewidth=2, ax=ax)
    ax.set_title('Distribution of trip length, weekday vs. weekend')
    ax.set_xlabel('Trip length (minutes, up to 90 shown)')
    ax.set_ylabel('Share of trips (density)')
    ax.get_legend().set_title('Day type')
    ax.get_legend().set_frame_on(False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_duration_box(trips, path):
    """Box plot of trip length by day type and user type."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    sns.boxplot(data=trips, x='usertype', y='minutes', hue='day_type',
                palette=DAY_TYPE_COLORS, showfliers=False, width=0.6,
                linewidth=1.2, ax=ax)
    ax.set_title('Trip length by day and rider type (outliers hidden)')
    ax.set_xlabel('Rider type')
    ax.set_ylabel('Trip length (minutes)')
    ax.legend(title='Day type', frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_neighborhoods(counts, path):
    """Horizontal bar chart of trips starting in each neighborhood."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ordered = counts.sort_values()
    ax.barh(ordered.index, ordered.values, color=WEEKDAY_COLOR, height=0.7)
    for i, value in enumerate(ordered.values):
        ax.text(value + 800, i, f'{value:,}', va='center', fontsize=8,
                color='#52514e')
    ax.set_title('Trips by starting neighborhood')
    ax.set_xlabel('Number of trips started')
    ax.set_xlim(0, ordered.max() * 1.15)
    ax.set_xticks([0, 20000, 40000, 60000, 80000])
    ax.grid(axis='x', color='#e6e5e1', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_top_stations(station_counts, path):
    """Horizontal bar chart of the 10 stations where the most trips start."""
    top = station_counts.head(10).sort_values('trips')
    labels = top['name'].str.split(' / ').str[-1] + ' (' + \
        top['neighborhood'] + ')'
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.barh(labels, top['trips'], color=WEEKDAY_COLOR, height=0.7)
    for i, value in enumerate(top['trips']):
        ax.text(value + 150, i, f'{value:,}', va='center', fontsize=8,
                color='#52514e')
    ax.set_title('Top 10 starting stations')
    ax.set_xlabel('Number of trips started')
    ax.set_xlim(0, top['trips'].max() * 1.2)
    ax.set_xticks([0, 5000, 10000])
    ax.grid(axis='x', color='#e6e5e1', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rq1_assumptions(daily, path):
    """
    Histograms of daily trip counts and daily mean temperature, used to
    check the normality assumption for the Pearson correlation test.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.6, 2.7))
    sns.histplot(daily['trip_count'], bins=30, color=WEEKDAY_COLOR,
                 ax=left)
    left.set_title('Trips per day')
    left.set_xlabel('Trips')
    sns.histplot(daily['Mean_Temperature_F'], bins=30,
                 color=WEEKEND_COLOR, ax=right)
    right.set_title('Mean temperature')
    right.set_xlabel('°F')
    style_axes(left)
    style_axes(right)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def check_rq1_assumptions(daily):
    """
    Checks the assumptions of a Pearson correlation between daily mean
    temperature and daily trip count. Returns a dict with the Shapiro-Wilk
    p-values, the correlation between each day's trip count and the day
    before's (to check independence), and the Pearson and
    Spearman correlation coefficients (as a preview, not the final test).
    """
    temp = daily['Mean_Temperature_F']
    count = daily['trip_count']
    return {
        'shapiro_p_trips': stats.shapiro(count).pvalue,
        'shapiro_p_temp': stats.shapiro(temp).pvalue,
        'day_to_day_corr': count[1:].reset_index(drop=True).corr(
            count[:-1].reset_index(drop=True)),
        'pearson_r': temp.corr(count),
        'spearman_rho': temp.corr(count, method='spearman'),
    }


def check_rq2_assumptions(trips):
    """
    Checks the assumptions of a two-sample t-test on trip length between
    weekdays and weekends. Returns a dict with group sizes, the skewness of
    each group, and the Levene test p-value for equal variances.
    """
    weekday = trips.loc[trips['day_type'] == 'Weekday', 'minutes']
    weekend = trips.loc[trips['day_type'] == 'Weekend', 'minutes']
    return {
        'n_weekday': len(weekday),
        'n_weekend': len(weekend),
        'skew_weekday': weekday.skew(),
        'skew_weekend': weekend.skew(),
        'levene_p': stats.levene(weekday, weekend).pvalue,
    }


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
        print('No missing values (df.isna().sum().sum() == 0)')
    else:
        print('Missing values:')
        print(report.round(2))


def main():
    sns.set_theme(style='white', font_scale=1.0)

    raw_lines = sum(1 for _ in open(DATA_DIR + '/trip.csv')) - 1
    trips = load_trips(DATA_DIR + '/trip.csv')
    weather = load_weather(DATA_DIR + '/weather.csv')
    stations = load_stations(DATA_DIR + '/station.csv')
    print('Data lines in trip.csv:', raw_lines)
    print('Trips loaded:', len(trips), '| skipped:', raw_lines - len(trips))
    raw_weather = pd.read_csv(DATA_DIR + '/weather.csv',
                              na_values='-')
    print('Raw weather missing values:')
    print(missing_report(raw_weather).round(2))
    print('Date range:', weather['Date'].iloc[0], 'to',
          weather['Date'].iloc[-1])

    # Dataset 1: daily trips joined with weather
    daily = daily_trip_weather(trips, weather)
    describe_dataset('DATASET 1: daily trips + weather', daily)
    print('Days with zero trips:', (daily['trip_count'] == 0).sum())
    print('\nSummary of trip_count, Mean_Temperature_F, Precipitation_In:')
    print(pd.DataFrame({
        'trip_count': seven_number_summary(daily['trip_count']),
        'Mean_Temperature_F':
            seven_number_summary(daily['Mean_Temperature_F']),
        'Precipitation_In': seven_number_summary(daily['Precipitation_In']),
    }).round(2))
    print('\nday_type counts (days):')
    print(daily['day_type'].value_counts())
    rq1 = check_rq1_assumptions(daily)
    print('\nRQ1 assumption checks:')
    for key in rq1:
        print(f'  {key}: {rq1[key]:.4g}')
    plot_temp_vs_trips(daily, FIG_DIR + '/temp_vs_trips.png')
    plot_trips_over_time(daily, FIG_DIR + '/trips_over_time.png')
    plot_rq1_assumptions(daily, FIG_DIR + '/rq1_hist.png')

    # Dataset 2: trips on their own
    describe_dataset('DATASET 2: trips', trips)
    print('\nSummary of minutes by day_type:')
    for day_type in ['Weekday', 'Weekend']:
        minutes = trips[trips['day_type'] == day_type]['minutes']
        print(day_type)
        print(seven_number_summary(minutes).round(2))
    print('\nday_type counts (trips):')
    print(trips['day_type'].value_counts())
    print('\nusertype counts:')
    print(trips['usertype'].value_counts())
    print('\nMedian minutes by usertype and day_type:')
    print(trips.groupby(['usertype', 'day_type'])['minutes'].median()
          .round(2))
    print('\nShare of trips by short-term pass holders, by day_type:')
    short_term = trips['usertype'] == 'Short-Term Pass Holder'
    print(short_term.groupby(trips['day_type']).mean().round(3))
    print('Trips over 3 hours:', (trips['minutes'] > 180).sum())
    rq2 = check_rq2_assumptions(trips)
    print('\nRQ2 assumption checks:')
    for key in rq2:
        print(f'  {key}: {rq2[key]:.4g}')
    plot_duration_hist(trips, FIG_DIR + '/duration_hist.png')
    plot_duration_box(trips, FIG_DIR + '/duration_box.png')

    # Dataset 3: trips joined with stations
    joined = trips_with_stations(trips, stations)
    describe_dataset('DATASET 3: trips + stations', joined)
    print('Trips dropped by join (not a real station):',
          len(trips) - len(joined))
    print('Unique starting stations:', joined['station_id'].nunique())
    print('\nTrips per neighborhood:')
    hoods = top_neighborhoods(joined)
    print(hoods)
    station_counts = trips_per_station(joined)
    print('\nTop 10 starting stations:')
    print(station_counts.head(10).to_string(index=False))
    print('\nSummary of trips per station:')
    print(seven_number_summary(station_counts['trips']).round(2))
    plot_neighborhoods(hoods, FIG_DIR + '/neighborhoods.png')
    plot_top_stations(station_counts,
                      FIG_DIR + '/top_stations.png')


if __name__ == '__main__':
    main()
