from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from common.challenge import Challenge


class CamelCardsPart1(Challenge):
    SAMPLE_RESULT = 6440
    VALUES = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    MAPPING = {val: num for num, val in enumerate(reversed(VALUES))}

    def calculate_hand(self, hand: str) -> pd.Series:
        counts = sorted([hand.count(card) for card in hand[0:5]], reverse=True)
        card_values = [self.MAPPING[card] for card in hand]
        return pd.Series(counts + card_values)

    def solve(self, data: pd.Series) -> int:
        data = data.str.split(" ", expand=True)

        scores = data[0].apply(self.calculate_hand)
        bids = pd.to_numeric(data[1]).rename("bid")

        hand_bid_scores = pd.concat([bids, scores], axis=1)
        hand_bid_scores = hand_bid_scores.sort_values(
            by=list(scores.columns),
        ).reset_index(drop=True)
        bids = hand_bid_scores["bid"]

        return (bids * (bids.index + 1)).sum()


class CamelCardsPart2(CamelCardsPart1):
    SAMPLE_RESULT = 5905
    VALUES = ["A", "K", "Q", "T", "9", "8", "7", "6", "5", "4", "3", "2", "J"]

    def calculate_hand(self, hand: str) -> pd.Series:
        values = "".join(set(hand))
        counts = sorted(
            [hand.count(card) if card != "J" else 0 for card in values], reverse=True
        )
        counts[0] += hand.count("J")
        card_values = [self.MAPPING[card] for card in hand]
        return pd.Series(counts[0:2] + card_values)
