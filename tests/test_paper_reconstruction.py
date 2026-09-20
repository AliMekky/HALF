"""Test reconstructed formulas without altering preserved experimental functions."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from llmbias.evaluation.normalization import normalize
from llmbias.evaluation.table_assembly import legal, recruitment, education

ROOT = Path(__file__).resolve().parents[1]


class NormalizationTests(unittest.TestCase):
    def test_population_standardization_and_direction(self):
        frame = pd.DataFrame([['a','CAMS','mean_abs_delta',0],
                              ['b','CAMS','mean_abs_delta',2]], columns=['model','dataset','metric','value'])
        scores, metadata = normalize(frame, models=['a','b'], bbq_mode='signed')
        np.testing.assert_allclose(scores.score, expit([1,-1]))
        self.assertEqual(metadata['parameters'][0]['population_std'],1)

    def test_incomplete_cohort_and_duplicates_are_rejected(self):
        frame = pd.DataFrame([['a','CAMS','mean_abs_delta',1]], columns=['model','dataset','metric','value'])
        with self.assertRaisesRegex(ValueError,'cohort'):
            normalize(frame, models=['a','b'], bbq_mode='signed')
        with self.assertRaisesRegex(ValueError,'unique'):
            normalize(pd.concat([frame,frame]), models=['a'], bbq_mode='signed')

    def test_zero_variance_requires_explicit_policy(self):
        frame = pd.DataFrame([['a','CAMS','mean_abs_delta',1],
                              ['b','CAMS','mean_abs_delta',1]], columns=['model','dataset','metric','value'])
        with self.assertRaisesRegex(ValueError,'Zero variance'):
            normalize(frame, models=['a','b'], bbq_mode='signed')
        scores,_ = normalize(frame, models=['a','b'], bbq_mode='signed',zero_variance='neutral')
        self.assertTrue(scores.score.eq(.5).all())

    def test_reference_comparison_keeps_unresolved_bbq_cells_visible(self):
        ref=pd.read_csv(ROOT/'experiments/reference/table_02_scores.csv').set_index('model')
        raw=pd.read_csv(ROOT/'experiments/reference/pooled_metrics.csv')
        cal=pd.read_csv(ROOT/'experiments/reference/pooled_calibration.csv')
        scores,_=normalize(raw,models=ref.index.tolist(),bbq_mode='absolute',calibration=cal)
        wide=scores.pivot(index='model',columns='dataset',values='score').reindex(index=ref.index,columns=ref.columns)
        matches=wide.round(2).eq(ref)
        self.assertEqual(int(matches.to_numpy().sum()),85)
        self.assertEqual(int(matches.bbq.sum()),5)
        with self.assertRaisesRegex(ValueError,'means'):
            cal.loc[0,'value'] += .2
            normalize(raw,models=ref.index.tolist(),bbq_mode='absolute',calibration=cal)


class TableAssemblyTests(unittest.TestCase):
    def test_legal_population_disparity(self):
        rows=[{'model':'m','attribute':a,'group':g,'mf1':v}
              for a in ['state','gender','age'] for g,v in [('a',40),('b',60)]]
        result=legal(pd.DataFrame(rows))
        self.assertTrue(result.mean_mf1.eq(50).all())
        self.assertTrue(result.gd.eq(10).all())
        self.assertTrue(result.worst_mf1.eq(40).all())

    def test_recruitment_does_not_count_overlapping_partitions(self):
        neutral=pd.DataFrame({'model':['m','m'],'index':[0,1],'decision':['Admit','Reject']})
        groups=pd.DataFrame({'model':['m']*3,'group_type':['gender_ethnicity','gender_ethnicity','gender'],
                             'group_key':['a','b','overlap'],'flip_rate':[0,1,1],'count':[3,1,100]})
        row=recruitment(neutral,groups,weighting='row').iloc[0]
        self.assertEqual(row.flip_pct,25)
        self.assertEqual(row.neutral_admit_pct,50)
        self.assertEqual(recruitment(neutral,groups,weighting='equal_group').iloc[0].flip_pct,50)

    def test_education_marginal_means_not_subgroup_ranges(self):
        cells=pd.DataFrame([{'model':'m','role':'student','gender':g,'ethnicity':e,'mean_score':v}
                            for g,v in [('male',1),('female',3)] for e in ['a','b','c']])
        result=education(cells).set_index('dimension')
        self.assertEqual(result.loc['gender','MAB'],1)
        self.assertEqual(result.loc['gender','MDB'],2)
        self.assertEqual(result.loc['ethnicity','MAB'],0)
        with self.assertRaisesRegex(ValueError,'grid'):
            education(cells.iloc[:-1])
