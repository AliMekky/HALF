"""Equation (1) reconstruction checks; these do not validate raw-metric normalization."""
import json
from pathlib import Path
import unittest

import pandas as pd

from llmbias.evaluation.aggregation import WEIGHTS, aggregate

ROOT = Path(__file__).resolve().parents[1]


class AggregationTests(unittest.TestCase):
    def test_displayed_table_two_scores_reproduce_all_published_totals(self):
        scores = pd.read_csv(ROOT / 'tests/fixtures/reference/table_02_scores.csv')
        expected = json.loads((ROOT / 'tests/fixtures/reference/table_02_totals.json').read_text())['totals']
        for row in aggregate(scores).to_dict('records'):
            self.assertEqual(round(row['naive'], 2), expected[row['model']]['naive'])
            self.assertEqual(round(row['half'], 2), expected[row['model']]['half'])
            self.assertEqual(row['total_weight'], 25)

    def test_missing_datasets_change_the_denominator(self):
        scores = pd.DataFrame([{'model': 'example', **dict.fromkeys(WEIGHTS, float('nan'))}])
        scores.loc[0, 'medbullets'] = .2
        scores.loc[0, 'ontonotes'] = .8
        result = aggregate(scores).iloc[0]
        self.assertAlmostEqual(result['half'], 35)
        self.assertEqual(result['datasets'], 2)
        self.assertEqual(result['total_weight'], 4)

    def test_invalid_or_empty_scores_are_rejected(self):
        scores = pd.DataFrame([{'model': 'example', **dict.fromkeys(WEIGHTS, .5)}])
        for invalid in [1.01, -.1, float('inf')]:
            scores.loc[0, 'medbullets'] = invalid
            with self.assertRaises(ValueError):
                aggregate(scores)
        scores.loc[0, list(WEIGHTS)] = float('nan')
        with self.assertRaises(ValueError):
            aggregate(scores)

    def test_duplicate_models_are_rejected(self):
        scores = pd.DataFrame([{'model': 'example', **dict.fromkeys(WEIGHTS, .5)}] * 2)
        with self.assertRaises(ValueError):
            aggregate(scores)
