from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import plotly.express as px

from common.challenge import Challenge


def brute_force_possibilities(time_to_beat: int, distance: int) -> int:
    time_to_charge = np.arange(1, int(time_to_beat))
    return np.sum(time_to_charge + distance / time_to_charge < time_to_beat)


class BoatRacePart1(Challenge):
    SAMPLE_RESULT = 288
    INDEX_DELIMITER = ":"

    def solve(self, data: pd.Series) -> int:
        data = data.str.split(" +", expand=True).T
        timing = pd.to_numeric(data["Time"], errors="coerce").dropna().to_numpy()
        distances = pd.to_numeric(data["Distance"], errors="coerce").dropna().to_numpy()

        total_possibilities = []
        for time_to_beat, distance in zip(timing, distances):
            total_possibilities.append(
                brute_force_possibilities(time_to_beat, distance)
            )
        return np.prod(total_possibilities)


class BoatRacePart2(BoatRacePart1):
    SAMPLE_RESULT = 71503

    def solve(self, data: pd.Series) -> int:
        data = data.str.replace(" ", "").T
        time_to_beat = int(data["Time"])
        distance = int(data["Distance"])

        return brute_force_possibilities(time_to_beat, distance)
