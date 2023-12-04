import pandas as pd

from common.challenge import Challenge

REGEX_DIGITS = r"(\d)"
DIGITS_AS_TEXT = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
REGEX_DIGITS_AND_TEXT = r"(?=(\d|" + "|".join(DIGITS_AS_TEXT) + "))"
REPLACE = {
    name: str(value)
    for value, name in enumerate(
        DIGITS_AS_TEXT,
        start=1,
    )
}


def to_calibration(data: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(
        data.groupby(level=0)[0].agg(
            lambda values: "".join([values.iloc[0], values.iloc[-1]])
        ),
        errors="raise",
    )


class TrebuchetPart1(Challenge):
    SAMPLE_RESULT = 142

    def solve(self, data: pd.Series) -> int:
        data = data.str.extractall(REGEX_DIGITS)
        data = to_calibration(data)
        return data.sum()


class TrebuchetPart2(Challenge):
    SAMPLE_RESULT = 281

    def solve(self, data: pd.Series) -> int:
        data = data.str.extractall(REGEX_DIGITS_AND_TEXT).replace(REPLACE)
        data = to_calibration(data)
        return data.sum()
