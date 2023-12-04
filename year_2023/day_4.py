from typing import Any, List, Tuple
import numpy as np
import pandas as pd

from common.challenge import Challenge


class ScratchCardsPart1(Challenge):
    SAMPLE_RESULT = 13
    INDEX_DELIMITER = ":"

    def parse(self, data: pd.Series) -> pd.DataFrame:
        all_values = data.str.split("|")
        winning = (
            all_values.apply(lambda values: values[0]).str.findall("\d+").rename(0)
        )
        numbers = (
            all_values.apply(lambda values: values[1]).str.findall("\d+").rename(1)
        )
        return pd.concat([winning, numbers], axis=1)

    def get_number_of_wins(self, data: pd.DataFrame) -> pd.Series:
        return data.apply(lambda row: len(set(row[0]).intersection(row[1])), axis=1)

    def solve(self, data: pd.Series) -> int:
        data = self.parse(data)
        data = self.get_number_of_wins(data).replace(0, pd.NA).dropna()
        return np.power(2, data - 1).sum()


class ScratchCardsPart2(ScratchCardsPart1):
    SAMPLE_RESULT = 30

    def solve(self, data: pd.Series) -> int:
        data = self.parse(data)
        data = self.get_number_of_wins(data)
        data.index = data.index.rename("card")
        data = pd.DataFrame(data.reset_index())
        data["total"] = pd.NA
        total = self.backwards_bonus(data, 0, len(data))
        return total

    def backwards_bonus(self, wins: pd.DataFrame, start: int, end: int) -> int:
        for idx in reversed(wins[start:end].index):
            bonus = 1
            n_wins = wins[0][idx]
            if n_wins > 0 and (idx + 1) < wins.shape[0]:
                bonus += wins["total"][idx + 1 : idx + n_wins + 1].sum()
            wins.loc[idx, "total"] = bonus
        return wins["total"].sum()
