# Seattle Bike Share and Weather (CSE 163 Final Project)

Aiden Hoang

## Files
- `eda.py` - Part 2 EDA. Loads and cleans the data, builds the joined datasets,
  prints every summary used in the report, and saves plots to `figures/`.
- `test_eda.py` - tests for `eda.py` using the small files in `test_data/`.
- `test_data/` - small hand-made CSV files used by the tests.
- `figures/` - plots made by `eda.py`.

## Data
The data files are not in this repo because they are large. Download the
Pronto Cycle Share dataset from
https://www.kaggle.com/datasets/pronto/cycle-share-dataset
and put `trip.csv`, `weather.csv`, and `station.csv` in a folder called `data/`
next to `eda.py`.

## How to run
```
python eda.py        # prints results, writes figures/
python test_eda.py   # runs the tests (the last test needs data/)
```
Requires pandas, matplotlib, seaborn, and scipy.
