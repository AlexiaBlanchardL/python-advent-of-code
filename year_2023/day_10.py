from typing import Any, Tuple, List
import numpy as np
import pandas as pd
from common.challenge import Challenge
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Connected:
    def __init__(self, data: np.ndarray, center_position: Tuple[int, int]) -> None:
        self.data = data
        self.center_position = center_position
        x = center_position[0]
        y = center_position[1]
        self.center = data[x, y]
        candidates = self.candidates()
        self.south = (
            candidates.get("south") if candidates.get("south", ".") in "|JL" else None
        )
        self.east = (
            candidates.get("east") if candidates.get("east", ".") in "-7J" else None
        )
        self.north = (
            candidates.get("north") if candidates.get("north", ".") in "|F7" else None
        )
        self.west = (
            candidates.get("west") if candidates.get("west", ".") in "-FL" else None
        )

    def candidates(self):
        if self.center == "-":
            return {"east": self.data[self.east_pos], "west": self.data[self.west_pos]}
        if self.center == "|":
            return {
                "south": self.data[self.south_pos],
                "north": self.data[self.north_pos],
            }
        if self.center == "F":
            return {
                "south": self.data[self.south_pos],
                "east": self.data[self.east_pos],
            }
        if self.center == "7":
            return {
                "south": self.data[self.south_pos],
                "west": self.data[self.west_pos],
            }
        if self.center == "L":
            return {
                "east": self.data[self.east_pos],
                "north": self.data[self.north_pos],
            }
        if self.center == "J":
            return {
                "north": self.data[self.north_pos],
                "west": self.data[self.west_pos],
            }
        if self.center == "S":
            return {
                "south": self.data[self.south_pos],
                "east": self.data[self.east_pos],
                "north": self.data[self.north_pos],
                "west": self.data[self.west_pos],
            }
        raise Exception("No candidates", self.center)

    def is_connected_to_start(self) -> bool:
        return "S" in self.candidates().values()

    def __repr__(self) -> str:
        return (
            str(self.center)
            + ": "
            + str([self.south, self.north, self.east, self.west])
        )

    @property
    def south_pos(self) -> Tuple[int, int]:
        return (self.center_position[0] + 1, self.center_position[1])

    @property
    def east_pos(self) -> Tuple[int, int]:
        return (self.center_position[0], self.center_position[1] + 1)

    @property
    def north_pos(self) -> Tuple[int, int]:
        return (self.center_position[0] - 1, self.center_position[1])

    @property
    def west_pos(self) -> Tuple[int, int]:
        return (self.center_position[0], self.center_position[1] - 1)


class PipeMazePart1(Challenge):
    SAMPLE_RESULT = 8

    def calculate_path(self, data: pd.DataFrame) -> List[Connected]:
        start_position = np.where(data == "S")
        start_row, start_col = list(zip(start_position[0], start_position[1]))[0]
        connected = Connected(data, (start_row, start_col))
        start_node = connected
        nodes = []
        nodes_positions = []
        while (
            not (len(nodes) > 1 and connected.is_connected_to_start())
        ) and connected.center_position not in nodes_positions:
            nodes.append(connected)
            nodes_positions.append(connected.center_position)
            if connected.south and connected.south_pos not in nodes_positions:
                connected = Connected(data, connected.south_pos)
            elif connected.east and connected.east_pos not in nodes_positions:
                connected = Connected(data, connected.east_pos)
            elif connected.north and connected.north_pos not in nodes_positions:
                connected = Connected(data, connected.north_pos)
            elif connected.west and connected.west_pos not in nodes_positions:
                connected = Connected(data, connected.west_pos)
            else:
                raise Exception("No connection")
        nodes.append(connected)
        nodes.append(start_node)
        return nodes

    def solve(self, data: pd.Series) -> Any:
        data = data.str.split("", expand=True).to_numpy()
        nodes = self.calculate_path(data)
        return int((len(nodes) - 1) / 2)


class PipeMazePart2(PipeMazePart1):
    SAMPLE_RESULT = 8

    def solve(self, data: pd.Series) -> Any:
        data = data.str.split("", expand=True).to_numpy()
        nodes = self.calculate_path(data)
        nodes_positions = [node.center_position for node in nodes]
        polygon = Polygon(nodes_positions)
        area = 0
        contour = np.zeros_like(data)
        x = [node[0] for node in nodes_positions]
        y = [node[1] for node in nodes_positions]
        contour[x, y] = 1
        temp = pd.DataFrame(contour).replace(0, np.nan)
        candidates = np.zeros_like(data)
        candidates[
            (temp.ffill() == 1)
            & (temp.bfill() == 1)
            & (temp.T.ffill().T == 1)
            & (temp.T.bfill().T == 1)
        ] = 1
        candidates = np.where(candidates - contour)
        for x, y in zip(candidates[0], candidates[1]):
            if polygon.contains(Point(x, y)):
                area += 1
        return area
