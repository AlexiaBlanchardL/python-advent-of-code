import inspect
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from termcolor import colored


class Challenge:
    SAMPLE_RESULT = None
    INDEX_DELIMITER = None

    def __init__(self) -> None:
        self._set_files()
        self._validate()
        self._test()

    def _set_files(self) -> None:
        challenge_file = Path(inspect.getfile(self.__class__))
        day = challenge_file.stem
        year = challenge_file.parent.stem
        part = self.__class__.__name__[-1]
        self.input_dir = os.path.join(year, "inputs")
        sample = f"{day}_sample_{part}.txt"
        input_name = f"{day}.txt"
        self.sample_filename = os.path.join(self.input_dir, sample)
        self.input_filename = os.path.join(self.input_dir, input_name)

    def _validate(self) -> None:
        assert os.path.exists(
            self.sample_filename
        ), f"Please save the sample input ({self.sample_filename}) in folder {self.input_dir}"
        assert os.path.exists(
            self.input_filename
        ), f"Please save the challenge input ({self.input_filename}) in folder {self.input_dir}"
        assert self.SAMPLE_RESULT is not None, "Please fill the sample result value."

    @classmethod
    def _to_series(cls, filename: str) -> pd.Series:
        data = pd.read_csv(
            filename,
            delimiter=cls.INDEX_DELIMITER,
            index_col=0 if cls.INDEX_DELIMITER else None,
            header=None,
            skip_blank_lines=False,
        )
        return data[data.columns[0]]

    def solve(self, data: pd.Series) -> Any:
        raise NotImplementedError("solve() method should be implemented.")

    def _test(self) -> None:
        data = self._to_series(self.sample_filename)
        result = self.solve(data)
        assert (
            result == self.SAMPLE_RESULT
        ), f"Solver should return {self.SAMPLE_RESULT}, returned {result}"
        print(colored(f"{self.__class__.__name__} passed test!", "green"))

    def response(self) -> None:
        data = self._to_series(self.input_filename)
        tic = time.perf_counter()
        response = self.solve(data)
        toc = time.perf_counter()
        print(colored("=== RESULT ===", "blue"))
        print(colored(response, "blue"))
        print(colored("==============", "blue"))
        timed = colored(f"{(toc-tic)*1000:.2f}", "yellow")
        print(f"{self.__class__.__name__} solved in {timed} ms")
        print("")
