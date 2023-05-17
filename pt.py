import csv
from datetime import datetime
from tqdm import tqdm
def changefile(): 
    with open("all_games.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter = ",")
        writer = csv.writer(open("pbp2122.csv", "w", encoding="utf-8"))
        writer.writerow(next(reader))
        for row in tqdm(reader): 
            curdate = datetime.strptime(row[2], "%M/%d/%Y")
            if (curdate.year >= 2014 and curdate.year <= 2020 and (curdate.year < 2020 or curdate.month != 12)):
                writer.writerow(row)




changefile()