from typing import List, Tuple, Dict
import pandas as pd

from common.challenge import Challenge


def split_into_numbers(value: str) -> List[int]:
    return [int(val) for val in value.split(" ")]


def is_in_range(value: int, range_start: int, range_len: int) -> bool:
    return range_start <= value <= range_start + range_len


def convert(value: int, source_start: int, destination_start) -> int:
    distance = value - source_start
    return destination_start + distance


def extract_mapping(data_after_seeds: pd.Series) -> Dict[str, List]:
    maps = {}
    last_map = None
    for line in data_after_seeds:
        if "map" in line:
            last_map = line
            maps[line] = []
        else:
            maps[last_map].append(split_into_numbers(line))
    return maps


class FertilizerPart1(Challenge):
    SAMPLE_RESULT = 35

    def to_location(self, maps, seed: int) -> int:
        converted = seed
        for mapping in maps.values():
            for destination, source, length in mapping:
                if is_in_range(converted, source, length):
                    converted = convert(converted, source, destination)
                    break
        return converted

    def solve(self, data: pd.Series) -> int:
        data = data.dropna()
        seeds = split_into_numbers(data[0].replace("seeds: ", ""))
        maps = extract_mapping(data_after_seeds=data[1:])
        locations = []
        for seed in seeds:
            locations.append(self.to_location(maps, seed))
        return min(locations)


class FertilizerPart2(Challenge):
    SAMPLE_RESULT = 46

    def solve(self, data: pd.Series) -> int:
        data = data.dropna()
        seed_ranges = split_into_numbers(data[0].replace("seeds: ", ""))
        maps = extract_mapping(data[1:])

        locations = []
        for i in range(int(len(seed_ranges) / 2)):
            # Create pairs of (start, length)
            seed = (seed_ranges[2 * i], seed_ranges[2 * i + 1])
            locations.append(self.to_location(maps, seed))
        return min(locations)

    def to_location(self, maps, seed: Tuple[int, int]) -> List[int]:
        converted = [seed]
        # Conversion steps -> seed -> soil -> ... -> locations
        for mapping in maps.values():
            converted_updated = []
            # For each seed or converted value pairs
            for value in converted:
                converted_updated += self.convert_step(mapping, value)
            converted = converted_updated
        # Only keep min
        return min(val[0] for val in converted)

    def convert_step(
        self, mapping: List[List[int]], value: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        # For each possible map range until value is empty
        converted_ranges = []
        for destination, source, length in mapping:
            partial_result, remaining = self.partial_conversion(
                value, (destination, source, length)
            )
            if partial_result:
                converted_ranges.append(partial_result)
            if remaining:
                value = remaining
            elif partial_result:  # No remaining -> Complete result
                value = None
                break
        if value or not converted_ranges:  # Remaining parts never mapped
            converted_ranges.append(value)
        return converted_ranges

    def partial_conversion(
        self, value_range: Tuple[int, int], mapping: List[int]
    ) -> Tuple[int | None, int | None]:
        value_start, value_length = value_range
        destination, source, mapping_length = mapping
        min_in_range = is_in_range(value_start, source, mapping_length)
        max_in_range = is_in_range(value_start + value_length, source, mapping_length)
        if min_in_range or max_in_range:
            converted_start = convert(value_start, source, destination)
            # Complete match
            if min_in_range and max_in_range:
                return (
                    converted_start,
                    value_length,
                ), None
            # Partial match (min)
            if min_in_range and not max_in_range:
                partial_length = destination + mapping_length - converted_start
                remaining_length = value_length - partial_length
                return (
                    (
                        converted_start,
                        partial_length,
                    ),
                    (value_start + partial_length, remaining_length),
                )
            # Partial match (max)
            if not min_in_range and max_in_range:
                partial_length = value_start + value_length - source
                remaining_length = source - value_start
                return (
                    destination,
                    partial_length,
                ), (value_start, remaining_length)
        # No match
        return None, None
