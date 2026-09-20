"""Preservation checks against the recovered complete BOLD script."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from llmbias.evaluation.conversational.bold_metrics import analyze
from llmbias.evaluation.conversational.bold import evaluate, summarize_domains

FIXTURES = Path(__file__).parent/'fixtures'
SOURCE = (FIXTURES/'bold_eval_original.py.txt').read_text()


class VaderStub:
    def polarity_scores(self, text):
        return {'compound':(len(text)%7 - 3)/10}


class ToxicityStub:
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash='test-only-revision'))

    def __call__(self, text):
        return [[{'label':'insult','score':0.99},{'label':'TOXIC','score':len(text)/1000}]]


@contextlib.contextmanager
def fake_dependencies(vader_factory=VaderStub, calls=None):
    parent=ModuleType('vaderSentiment');vader=ModuleType('vaderSentiment.vaderSentiment')
    vader.SentimentIntensityAnalyzer=vader_factory
    transformers=ModuleType('transformers')
    def pipeline(*args,**kwargs):
        if calls is not None:calls.append((args,kwargs))
        return ToxicityStub()
    transformers.pipeline=pipeline
    with patch.dict(sys.modules,{'vaderSentiment':parent,'vaderSentiment.vaderSentiment':vader,'transformers':transformers}):yield


def oracle(records, response_format='openai', id_format='bold', vader_factory=VaderStub):
    source=SOURCE
    active='r.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")'
    alternatives={'anthropic':'r.get("result", {}).get("message", {}).get("content", [{}])[0].get("text", "")',
                  'deepseek':'r.get("response", "")'}
    if response_format != 'openai':source=source.replace('gen = '+active,'gen = '+alternatives[response_format])
    if id_format == 'short':source=source.replace('r"bold-([^-]+)-"','r"request-\\d+-(.+?)-"')
    with tempfile.TemporaryDirectory() as folder, fake_dependencies(vader_factory), contextlib.redirect_stdout(io.StringIO()):
        previous=Path.cwd()
        try:
            os.chdir(folder)
            Path('Meta-Llama-3.1-8B-Instruct_bold.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            exec(compile(source,'<original-conv_ai/eval.py>','exec'),{})
            return Path('domain_metrics.csv').read_bytes(),Path('gender_polarity.csv').read_bytes()
        finally:os.chdir(previous)


class OriginalBoldTests(unittest.TestCase):
    def records(self):
        return [{'custom_id':cid,'response':{'body':{'choices':[{'message':{'content':text}}]}}}
                for cid,text in [('request-0-test-bold-American_actors-person','He is a father and brother.'),
                                 ('request-1-test-bold-American_actresses-person','She helps her mother.'),
                                 ('request-2-test-bold-science-person','A lovely discovery.'),
                                 ('request-3-test-bold-science-person',''),('no-domain','Nothing here.')]]

    def test_full_source_csv_equivalence(self):
        expected=oracle(self.records())
        frames=analyze(self.records(),VaderStub(),ToxicityStub())
        self.assertEqual(expected,tuple(f.to_csv(index=False).encode() for f in frames))
        self.assertIn('unknown',set(frames[0].domain))
        self.assertAlmostEqual(frames[0].loc[frames[0].domain=='American_actors','avg_toxicity'].iloc[0],0.027)

    def test_commented_provider_alternatives_preserved(self):
        for provider in ['anthropic','deepseek']:
            records=[]
            for entry in self.records()[:3]:
                cid=entry['custom_id'].replace('-test-bold-','-')
                text=entry['response']['body']['choices'][0]['message']['content']
                record={'custom_id':cid}
                if provider=='deepseek':record['response']=text
                else:record['result']={'message':{'content':[{'text':text}]}}
                records.append(record)
            with self.subTest(provider=provider):
                expected=oracle(records,provider,'short')
                actual=analyze(records,VaderStub(),ToxicityStub(),provider,'short')
                self.assertEqual(expected,tuple(f.to_csv(index=False).encode() for f in actual))

    def test_original_functions_are_preserved(self):
        from llmbias.evaluation.conversational import bold_metrics as bold_original
        moved=ast.parse(Path(bold_original.__file__).read_text())
        fn=next(n for n in moved.body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
        definitions={n.name:n for n in fn.body if isinstance(n,ast.FunctionDef)}
        for node in ast.parse(SOURCE).body:
            if isinstance(node,ast.FunctionDef):self.assertEqual(ast.dump(node),ast.dump(definitions[node.name]))

    def test_missing_toxic_label_fallback(self):
        frames=analyze(self.records(),VaderStub(),lambda text:[ [{'label':'other','score':0.9}] ])
        self.assertEqual(frames[0].avg_toxicity.tolist(),[0.0]*4)

    def test_default_model_and_outputs_match_script(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);input_path=root/'input.jsonl'
            input_path.write_text(''.join(json.dumps(r)+'\n' for r in self.records()))
            calls=[]
            with fake_dependencies(calls=calls):result=evaluate(input_path,root/'output')
            self.assertEqual(calls,[(('text-classification',),{'model':'unitary/toxic-bert','return_all_scores':True})])
            expected=oracle(self.records())
            self.assertEqual((root/'output/domain_metrics.csv').read_bytes(),expected[0])
            self.assertEqual((root/'output/gender_polarity.csv').read_bytes(),expected[1])
            self.assertEqual(result['groups'],4)
            manifest=json.loads((root/'output/provenance.json').read_text())
            self.assertEqual(manifest['toxicity']['resolved_revision'],'test-only-revision')
            with self.assertRaises(FileExistsError):evaluate(input_path,root/'output')

    def test_request_batches_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'requests.jsonl'
            source.write_text(json.dumps({'method':'POST','url':'/v1/chat/completions','body':{}})+'\n')
            calls=[]
            with fake_dependencies(calls=calls),self.assertRaisesRegex(ValueError,'generation requests'):
                evaluate(source,root/'output')
            self.assertEqual(calls,[])
            self.assertFalse((root/'output').exists())

    def test_saved_summary_agrees_with_paper_rounding(self):
        summary=summarize_domains(pd.read_csv(FIXTURES/'bold_saved_domain_metrics.csv'))
        self.assertEqual(summary['count'],1000)
        self.assertEqual(summary['groups'],41)
        self.assertEqual(round(summary['avg_sentiment'],3),0.124)
        self.assertEqual(round(summary['avg_toxicity']*1000,2),1.20)

    def test_default_workflow_is_original(self):
        from llmbias._workflows import WORKFLOWS
        self.assertEqual(WORKFLOWS['evaluate-bold'].target,'evaluation.conversational.bold:evaluate')
        self.assertIn('conv_ai/eval.py',WORKFLOWS['evaluate-bold'].source)
        self.assertNotIn('evaluate-bold-reference', WORKFLOWS)

    @unittest.skipUnless(importlib.util.find_spec('vaderSentiment'),'Install bold extra for real VADER preservation check')
    def test_real_vader_matches_original_script(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        expected=oracle(self.records(),vader_factory=SentimentIntensityAnalyzer)
        actual=analyze(self.records(),SentimentIntensityAnalyzer(),ToxicityStub())
        self.assertEqual(expected,tuple(f.to_csv(index=False).encode() for f in actual))


if __name__=='__main__':unittest.main()
