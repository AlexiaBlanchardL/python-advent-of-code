from typing import Any
import pandas as pd

from common.challenge import Challenge

REGEX_BLUE = r"(\d+) blue"
REGEX_GREEN = r"(\d+) green"
REGEX_RED = r"(\d+) red"
REGEX_GAME = r"Game (\d+)"

MAX_BLUE = 14
MAX_GREEN = 13
MAX_RED = 12


def extract_number(data: pd.Series, regex: str) -> pd.Series:
    return pd.to_numeric(
        data.str.extractall(regex)[0],
        errors="coerce",
    )


class CubeConundrumPart1(Challenge):
    SAMPLE_RESULT = 8
    INDEX_DELIMITER = ":"

    def extract_games_set(self, data: pd.Series) -> pd.DataFrame:
        data = data.str.split(";").explode()
        games = extract_number(pd.Series(data.index), REGEX_GAME).rename("game")
        data = data.reset_index()[1]

        blues = extract_number(data, REGEX_BLUE).rename("blue")
        greens = extract_number(data, REGEX_GREEN).rename("green")
        reds = extract_number(data, REGEX_RED).rename("red")

        return pd.concat([games, blues, greens, reds], axis=1).fillna(0)

    def solve(self, data: pd.Series) -> Any:
        data = self.extract_games_set(data)

        possible = (
            (data["blue"] <= MAX_BLUE)
            & (data["green"] <= MAX_GREEN)
            & (data["red"] <= MAX_RED)
        )
        possible = possible.groupby(by=data["game"]).agg(all)
        return sum(possible.index[possible])


class CubeConundrumPart2(CubeConundrumPart1):
    SAMPLE_RESULT = 2286

    def solve(self, data: pd.Series) -> Any:
        data = self.extract_games_set(data)

        min_red = data["red"].groupby(by=data["game"]).max()
        min_blue = data["blue"].groupby(by=data["game"]).max()
        min_green = data["green"].groupby(by=data["game"]).max()
        return (min_red * min_blue * min_green).sum()
