# stats100-proj

This repository contains source code for a course project from Stanford's Spring 2023 offering of Stats 100: Mathematics of Sports.
To learn more about the project, read our [blog post](https://stats100blog.wordpress.com/2023/06/09/predicting-nba-possession-outcomes-using-regression).

To begin, download NBA play-by-play data from [Kaggle](https://www.kaggle.com/datasets/xocelyk/nba-pbp). To select games, run `filter_games.py`.

By default, dates in the file are set to select games from the 2018-19 NBA regular season.

Next, use `get_player_stats.ipynb` to get player stats from the season. Use `compute_rolling_stats.ipynb` to get each players points/assists/rebounds per minute up to each date they played.

Download RAPTOR data from [FiveThirtyEight](https://projects.fivethirtyeight.com/nba-player-ratings) and use `raptor_name2id.ipynb` to add a BasketballReference name ID column to the RAPTOR data.

Combine these three datasets into one possession-by-possession dataset using `combine_features.ipynb`.

Now you can train models using `linear_reg.py` and `logistic_reg.py`.

To see our analysis, check `plot.ipynb`.
