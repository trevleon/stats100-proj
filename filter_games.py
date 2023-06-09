import csv
from datetime import datetime


def filter_games(in_file, out_file, start_date, end_date):
    with open(in_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        writer = csv.writer(open(out_file, "w", encoding="utf-8"))
        writer.writerow(next(reader))
        for row in reader:
            cur_date = datetime.strptime(row[2], "%m/%d/%Y")
            if (cur_date >= start_date and cur_date <= end_date):
                writer.writerow(row)


# all_games.csv downloaded from https://www.kaggle.com/datasets/xocelyk/nba-pbp
filter_games(in_file="all_games.csv",
             out_file="pbp_18-19.csv",
             start_date=datetime(2018, 10, 16),
             end_date=datetime(2019, 4, 10))
