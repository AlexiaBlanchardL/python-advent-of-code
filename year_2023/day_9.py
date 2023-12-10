from typing import Any
import numpy as np
import pandas as pd
from common.challenge import Challenge


class MiragePart1(Challenge):
    SAMPLE_RESULT = 114

    def extrapolate(self, values: np.ndarray) -> int:
        a = values[0:-1]
        b = values[1:]
        diffs = []
        while sum(abs(b - a)) > 0:
            diff = b - a
            diffs.append(diff[-1])
            a = diff[0:-1]
            b = diff[1:]
        return values[-1] + sum(diffs)

    def solve(self, data: pd.Series) -> Any:
        total = 0
        for row in data:
            values = pd.to_numeric(row.split(" "))
            total += self.extrapolate(values)
        return total


class MiragePart2(MiragePart1):
    SAMPLE_RESULT = 2

    def solve(self, data: pd.Series) -> Any:
        total = 0
        for row in data:
            values = pd.to_numeric(row.split(" "))[::-1]
            total += self.extrapolate(values)
        return total
