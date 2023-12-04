import re
import time
from typing import Any, List, Tuple
import numpy as np
import pandas as pd

from common.challenge import Challenge

REGEX_IS_SYMBOL = r"[^\d.]"
REGEX_CAPTURE_NUMBERS = r"(\d+)"
REGEX_GEAR = r"\*"

EXACT_NUMBER_OF_PARTS = 2


def matches(row):
    return [
        (m.group(0), m.start() - 1, m.end())
        for m in re.finditer(REGEX_CAPTURE_NUMBERS, row)
    ]


def get_all_numbers_and_positions(data: pd.Series) -> pd.DataFrame:
    numbers = []
    numbers = data.apply(matches).explode().dropna()
    numbers = pd.DataFrame(
        numbers.tolist(), index=numbers.index, columns=["number", "col_min", "col_max"]
    ).astype("Int64")
    return numbers.reset_index(names=["row"]).to_numpy()


def get_positions_of(data: pd.DataFrame, regex: str):
    return (
        data.apply(lambda col: col.str.match(regex).astype("Int64"), axis=1)
        .to_numpy()
        .nonzero()
    )


def numbers_neighbors_of(
    positions: List[np.ndarray], numbers: np.ndarray
) -> np.ndarray:
    # Shape for calculations = N x M
    shape = numbers.shape[0], positions[0].shape[0]
    # Tile positions M x 1 -> N x M
    positions_rows = np.tile(positions[0], (shape[0], 1))
    positions_cols = np.tile(positions[1], (shape[0], 1))
    # Tile numbers N x 1 -> N x M
    rows = np.tile(numbers[:, 0], (shape[1], 1)).T
    number = np.tile(numbers[:, 1], (shape[1], 1)).T
    col_min = np.tile(numbers[:, 2], (shape[1], 1)).T
    col_max = np.tile(numbers[:, 3], (shape[1], 1)).T
    # Check if numbers are adjacent to positions all at once
    adjacent_rows = np.abs(rows - positions_rows) <= 1
    adjacent_cols = (col_min <= positions_cols) & (positions_cols <= col_max)
    # Filter
    numbers_to_keep = number[adjacent_cols & adjacent_rows]
    return numbers_to_keep


class GearRatiosPart1(Challenge):
    SAMPLE_RESULT = 4361

    def solve(self, data: pd.Series) -> int:
        positions_of_symbols = get_positions_of(
            data.apply(lambda x: pd.Series(list(x))), REGEX_IS_SYMBOL
        )
        numbers = get_all_numbers_and_positions(data)
        numbers_to_keep = numbers_neighbors_of(positions_of_symbols, numbers)
        return sum(numbers_to_keep)


class GearRatiosPart2(Challenge):
    SAMPLE_RESULT = 467835

    def solve(self, data: pd.Series) -> int:
        positions_of_gears = get_positions_of(
            data.apply(lambda x: pd.Series(list(x))), REGEX_GEAR
        )
        numbers = get_all_numbers_and_positions(data)

        numbers_to_keep = []
        for row_gear, col_gear in zip(positions_of_gears[0], positions_of_gears[1]):
            gear_numbers = numbers_neighbors_of(
                np.array([[row_gear], [col_gear]]), numbers
            )
            if len(gear_numbers) == EXACT_NUMBER_OF_PARTS:
                numbers_to_keep.append(np.prod(gear_numbers))

        return sum(numbers_to_keep)
