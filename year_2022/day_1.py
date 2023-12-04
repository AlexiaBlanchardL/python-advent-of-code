import numpy as np
import pandas as pd
from typing import List

from common.challenge import Challenge

KEEP_TOP = 3


def total_per_elf(data: pd.Series) -> List[int]:
    data = data.astype("Int64")
    changepoints = data.index[data.isna()]
    per_elf = np.array_split(data.fillna(0).to_numpy(), changepoints)
    calories = []
    for calories_elf in per_elf:
        calories.append(calories_elf.sum())
    return calories


class CalorieCountingPart1(Challenge):
    SAMPLE_RESULT = 24000

    def solve(self, data: pd.Series) -> int:
        calories = total_per_elf(data)
        return max(calories)


class CalorieCountingPart2(Challenge):
    SAMPLE_RESULT = 45000

    @staticmethod
    def _to_series(filename: str) -> pd.Series:
        return pd.read_csv(
            filename, delimiter=None, header=None, skip_blank_lines=False
        )[0]

    def solve(self, data: pd.Series) -> int:
        calories = total_per_elf(data)
        calories = sorted(calories, reverse=True)
        return sum([calories[i] for i in range(KEEP_TOP)])
