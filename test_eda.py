"""
Aiden Hoang
CSE 163 Final Project, Part 2: Exploratory Data Analysis

Tests for eda.py. Every function is run on small hand-made CSV files
(trip_small.csv, weather_small.csv, station_small.csv) where the right
answers were worked out by hand. The small trip file includes one
malformed line (too many fields) and one trip from the Pronto repair
shop, which are the two data problems found in the real trip.csv.
"""

import math

import pandas as pd

import eda


TRIPS = 'trip_small.csv'
WEATHER = 'weather_small.csv'
STATIONS = 'station_small.csv'


def close(a, b):
    """Returns True if two floats are equal up to rounding error."""
    return math.isclose(a, b, abs_tol=1e-6)


def test_load_trips():
    """Checks bad-line skipping, minutes, dates, and weekday/weekend."""
    trips = eda.load_trips(TRIPS)
    # line with extra fields (trip 5) is skipped
    assert len(trips) == 4
    assert list(trips['trip_id']) == [1, 2, 3, 4]
    assert list(trips['minutes']) == [10, 20, 30, 5]
    # Oct 13 2014 was a Monday, Oct 18 a Saturday, Oct 19 a Sunday
    assert list(trips['day_type']) == ['Weekday', 'Weekday',
                                       'Weekend', 'Weekend']
    assert trips.loc[0, 'date'] == '10/13/2014'
    print('test_load_trips passed')


def test_load_weather():
    """Checks the filled temperature, '-' handling, and Events filling."""
    weather = eda.load_weather(WEATHER)
    assert len(weather) == 4
    # missing mean on 10/18 becomes (64 + 56) / 2
    assert close(weather.loc[1, 'Mean_Temperature_F'], 60)
    assert weather['Mean_Temperature_F'].isna().sum() == 0
    assert pd.isna(weather.loc[1, 'Max_Gust_Speed_MPH'])
    assert list(weather['Events']) == ['Rain', 'None', 'None', 'Rain']
    assert list(weather['month']) == ['2014-10'] * 4
    assert list(weather['day_type']) == ['Weekday', 'Weekend',
                                         'Weekend', 'Weekday']
    print('test_load_weather passed')


def test_load_stations():
    """Checks that station id codes map to neighborhood names."""
    stations = eda.load_stations(STATIONS)
    assert list(stations['neighborhood']) == ['Capitol Hill',
                                              'South Lake Union',
                                              'Downtown']
    print('test_load_stations passed')


def test_daily_trip_weather():
    """Checks trip counts per day, including a day with zero trips."""
    trips = eda.load_trips(TRIPS)
    weather = eda.load_weather(WEATHER)
    daily = eda.daily_trip_weather(trips, weather)
    assert len(daily) == 4
    assert list(daily['trip_count']) == [2, 1, 1, 0]
    assert list(daily['day_type']) == ['Weekday', 'Weekend',
                                       'Weekend', 'Weekday']
    # the total number of trips is kept by the join
    assert daily['trip_count'].sum() == len(trips)
    print('test_daily_trip_weather passed')


def test_trips_with_stations():
    """Checks that the shop trip is dropped and neighborhoods attach."""
    trips = eda.load_trips(TRIPS)
    stations = eda.load_stations(STATIONS)
    joined = eda.trips_with_stations(trips, stations)
    assert len(joined) == 3
    assert 'Pronto shop' not in list(joined['from_station_id'])
    hoods = eda.top_neighborhoods(joined)
    assert hoods['Capitol Hill'] == 2
    assert hoods['South Lake Union'] == 1
    assert hoods.index[0] == 'Capitol Hill'
    per_station = eda.trips_per_station(joined)
    assert list(per_station['name']) == ['A St', 'B St']
    assert list(per_station['trips']) == [2, 1]
    print('test_trips_with_stations passed')


def test_missing_report():
    """Checks missing counts and percents on the small trip file."""
    trips = eda.load_trips(TRIPS)
    report = eda.missing_report(trips)
    assert list(report.index) == ['gender', 'birthyear']
    assert list(report['missing']) == [2, 2]
    assert close(report.loc['gender', 'percent'], 50)
    no_missing = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert len(eda.missing_report(no_missing)) == 0
    print('test_missing_report passed')


def test_check_rq1_assumptions():
    """Checks the day-to-day correlation on a tiny made-up daily table."""
    daily = pd.DataFrame({'trip_count': [1, 2, 3, 4, 5, 6, 7, 8],
                          'Mean_Temperature_F': [40, 42, 41, 45, 50, 52,
                                                 51, 55]})
    result = eda.check_rq1_assumptions(daily)
    # trip counts go up by 1 every day, so each day is perfectly
    # correlated with the day before
    assert close(result['day_to_day_corr'], 1)
    print('test_check_rq1_assumptions passed')


def test_seven_number_summary():
    """Checks the summary against values worked out by hand."""
    summary = eda.seven_number_summary(pd.Series([1, 2, 3, 4, 5]))
    assert close(summary['mean'], 3)
    assert close(summary['std'], math.sqrt(2.5))
    assert close(summary['min'], 1)
    assert close(summary['Q1'], 2)
    assert close(summary['median'], 3)
    assert close(summary['Q3'], 4)
    assert close(summary['max'], 5)
    print('test_seven_number_summary passed')


def test_full_data_totals():
    """
    Checks that the real joined datasets agree with each other: no trips
    are lost when counting per day, and only the 5 shop/operations trips
    are lost in the station join.
    """
    trips = eda.load_trips('data/trip.csv')
    weather = eda.load_weather('data/weather.csv')
    stations = eda.load_stations('data/station.csv')
    daily = eda.daily_trip_weather(trips, weather)
    joined = eda.trips_with_stations(trips, stations)
    assert len(daily) == len(weather) == 689
    assert daily['trip_count'].sum() == len(trips)
    assert len(trips) - len(joined) == 5
    assert joined['neighborhood'].isna().sum() == 0
    print('test_full_data_totals passed')


def main():
    test_load_trips()
    test_load_weather()
    test_load_stations()
    test_daily_trip_weather()
    test_trips_with_stations()
    test_missing_report()
    test_check_rq1_assumptions()
    test_seven_number_summary()
    test_full_data_totals()
    print('All tests passed')


if __name__ == '__main__':
    main()
