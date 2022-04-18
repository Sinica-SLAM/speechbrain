import datetime
import json
import pickle
from typing import Dict


class DateTokenizer:
    def __init__(self, load_path: str = None):
        self.start_date_map = {}
        self.month_map = {}
        if load_path is not None:
            self.load(load_path)

    def train(self, data: Dict[str, Dict[str, str]]):
        start_dates = []
        for line in data.values():
            date_str = line["date"]
            date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            start_date = get_start_date(date)
            if start_date not in start_dates:
                start_dates.append(start_date)
        for i, start_date in enumerate(sorted(start_dates)):
            self.start_date_map[start_date] = i + 1

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            pickle.dump(self.start_date_map, f)

    def load(self, load_path: str):
        with open(load_path, "rb") as f:
            self.start_date_map = pickle.load(f)
            self.get_month_map()

    def get_month_map(self):
        for date, value in self.start_date_map.items():
            year = date.year
            month = date.month
            year_month = f'{year}{month:02}'
            if year_month in self.month_map:
                self.month_map[year_month].append(value)
            else:
                self.month_map[year_month] = [value]

    def encode(self, date: datetime.date) -> int:
        start_date = get_start_date(date)
        # if start_date not in self.start_date_map:
            # return len(self.start_date_map)
        return self.start_date_map[start_date]

    def encode_year_month(self, year_month:str):
        # year_month like 202101
        return self.month_map[year_month]
        


def get_start_date(date: datetime.date) -> datetime.date:
    week_day = date.isoweekday()
    return date - datetime.timedelta(days=(week_day + 2) % 7)


if __name__ == "__main__":
    data_file_path = "results/prepare_cna/train.json"
    tokenizer = DateTokenizer()
    tokenizer.train(json.load(open(data_file_path, "r", encoding="utf8")))
    tokenizer.save("results/date_tokenizer/date_tokenizer.model")
    print(len(tokenizer.start_date_map))
