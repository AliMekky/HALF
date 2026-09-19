"""Offline BOLD reference and integration checks; no model downloads or API calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock
import unittest
from unittest.mock import patch

import pandas as pd

from llmbias.evaluation.bold_reference import (LABELS, aggregate, evaluate, load_cached_scores,
                                    reduce_toxicity, score_local_bert)
from llmbias.parsing.bold import (DOMAINS, anonymize, assemble_text, process_file,
                                 read_jsonl, response_text, text_hash)

HAS_VADER = importlib.util.find_spec('vaderSentiment') is not None


class BoldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset.csv'
        self.raw = self.root/'raw.jsonl'
        self.normalized = self.root/'normalized.jsonl'
        self.cache = self.root/'toxicity.jsonl'
        rows, responses = [], []
        for i, domain_file in enumerate([*DOMAINS, 'profession_prompt.json']):
            prompt = f'Example {i} is '
            category, subject = f'category-{i}', f'Subject-{i}'
            rows.append({'prompt': prompt, 'file': domain_file, 'category': category, 'subcategory':subject})
            responses.append({'custom_id':f'request-{i}-wrong-request-model-bold-{category}-{subject}',
                              'response':{'status_code':200, 'body':{'model':'actual-model', 'choices':[
                                  {'finish_reason':'stop','message':{'content':'a wonderful success.'}}]}}})
        pd.DataFrame(rows).to_csv(self.dataset,index=False)
        self.write(self.raw, list(reversed(responses)))
        process_file(self.raw,self.dataset,self.normalized,'study-model','full_text','none')
        self.records = read_jsonl(self.normalized)
        self.write(self.cache,[{'custom_id':r['custom_id'],'text_sha256':r['text_sha256'],
                              'scores':dict.fromkeys(LABELS,0.1 + i*0.1)} for i,r in enumerate(self.records)])

    def write(self,path,records):
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_join_uses_index_and_actual_model_not_file_order_or_id_model(self):
        self.assertEqual([r['index'] for r in self.records],list(range(6)))
        self.assertEqual({r['source_model'] for r in self.records},{'actual-model'})
        self.assertEqual({r['model'] for r in self.records},{'study-model'})
        self.assertEqual(self.records[0]['text'],'Example 0 is a wonderful success.')

    def test_anthropic_text_blocks_ignore_thinking(self):
        obj={'result':{'type':'succeeded','message':{'model':'claude','stop_reason':'end_turn',
             'content':[{'type':'thinking','thinking':'private reasoning'},{'type':'text','text':'Hello'},
                        {'type':'text','text':' world'}]}}}
        self.assertEqual(response_text(obj),('Hello world','claude'))

    def test_errors_and_truncation_are_not_scored_as_neutral(self):
        for obj in [{'error':{'message':'failed'}},{'result':{'type':'errored'}},
                    {'response':{'status_code':429}},
                    {'response':{'body':{'choices':[{'finish_reason':'length','message':{'content':'x'}}]}}},
                    {'response':{'body':{'choices':[{'message':{'content':'','refusal':'no'}}]}}}]:
            with self.subTest(obj=obj),self.assertRaises(ValueError):response_text(obj)

    def test_assembly_does_not_duplicate_an_echoed_prompt(self):
        self.assertEqual(assemble_text('A teacher is ','A teacher is helpful.','full_text'),'A teacher is helpful.')
        self.assertEqual(assemble_text('A teacher is ','helpful.','response'),'helpful.')
        with self.assertRaises(ValueError):assemble_text('x','y','guess')

    def test_anonymization_is_explicit_and_boundary_aware(self):
        self.assertEqual(anonymize('Jane Doe and Jane study Janeville.',['Jane','Jane Doe'],'gender'),
                         'Person and Person study Janeville.')
        self.assertEqual(anonymize('A nurse helps.',['nurse'],'profession'),'A XYZ helps.')
        entities=self.root/'entities.json';entities.write_text(json.dumps({str(i):[f'Example {i}'] for i in range(6)}))
        out=self.root/'masked.jsonl'
        process_file(self.raw,self.dataset,out,'model','full_text','explicit',entities)
        self.assertEqual(read_jsonl(out)[1]['text'],'Person is a wonderful success.')
        entities.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Missing reviewed'):process_file(self.raw,self.dataset,self.root/'bad','m','full_text','explicit',entities)

    def test_missing_duplicates_and_mismatched_metadata_fail(self):
        original=read_jsonl(self.raw)
        for records in [original[:-1],original+[original[0]],
                        [{**original[0],'custom_id':'request-0-bold-wrong-subject'}]+original[1:]]:
            self.write(self.raw,records)
            with self.assertRaises(ValueError):process_file(self.raw,self.dataset,self.root/'bad','m','response','none')
            self.assertFalse((self.root/'bad').exists())

    def test_request_override_and_sanitized_ids_are_explicit(self):
        rows=pd.read_csv(self.dataset,keep_default_na=False)
        rows.loc[0,'subcategory']='Subject.0'
        rows.loc[0,'prompt']=''
        rows.to_csv(self.dataset,index=False)
        records=read_jsonl(self.raw)
        for record in records:
            if record['custom_id'].startswith('request-0-'):
                record['custom_id']='request-0-category-0-Subject0'
        self.write(self.raw,records)
        override=self.root/'override.json'
        override.write_text(json.dumps({'0':{'prompt':'A repaired prompt ', 'source':'synthetic saved request'}}))
        batch=self.root/'batch.jsonl'
        self.write(batch,[{'custom_id':f"request-{i}-category-{i}-"+('Subject.0' if i==0 else f'Subject-{i}'),
                          'body':{'messages':[{}, {'content':'A repaired prompt ' if i==0 else f'Example {i} is '}]}}
                         for i in range(6)])
        with self.assertRaisesRegex(ValueError,'metadata'):
            process_file(self.raw,self.dataset,self.root/'exact','m','response','none',metadata_overrides_path=override)
        path=self.root/'adapted'
        process_file(self.raw,self.dataset,path,'m','full_text','none',metadata_overrides_path=override,
                     id_policy='anthropic_sanitized',batch_path=batch)
        out=read_jsonl(path)[0]
        self.assertEqual(out['text'],'A repaired prompt a wonderful success.')
        self.assertEqual(out['metadata_override']['source'],'synthetic saved request')
        self.assertTrue(out['conversion_sources']['batch'])
        self.assertEqual(pd.read_csv(self.dataset,keep_default_na=False).iloc[0]['prompt'],'')

    @unittest.skipUnless(HAS_VADER,'Install the bold extra for upstream VADER integration')
    def test_sentiment_only_has_no_invented_toxicity(self):
        from llmbias.evaluation.bold_reference import evaluate_sentiment
        result=evaluate_sentiment(self.normalized,self.root/'sentiment')
        self.assertFalse(result['toxicity_available'])
        self.assertNotIn('avg_toxicity',result)
        self.assertNotIn('toxicity',pd.read_csv(self.root/'sentiment/per_response.csv').columns)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Install torch to test checkpoint adapter inference')
    def test_checkpoint_adapter_preserves_multilabel_sigmoid_and_tokenization(self):
        import torch
        config=SimpleNamespace(classifier_dropout=0.1,hidden_dropout_prob=0.1,model_type='bert',
            hidden_size=1024,num_hidden_layers=24,num_attention_heads=16,num_labels=6,
            id2label=dict(enumerate(LABELS)))
        model=MagicMock();model.config=config;model.to.return_value=model;model.eval.return_value=model
        model.return_value=SimpleNamespace(logits=torch.tensor([[0.,1.,-1.,0.,0.,0.]]))
        tokenizer=MagicMock();tokenizer.do_lower_case=True
        tokenizer.return_value={'input_ids':torch.ones((1,256),dtype=torch.long)}
        model_class=MagicMock();model_class.from_pretrained.return_value=(model,{})
        tokenizer_class=MagicMock();tokenizer_class.from_pretrained.return_value=tokenizer
        fake=SimpleNamespace(BertForSequenceClassification=model_class,BertTokenizer=tokenizer_class)
        with patch.dict('sys.modules',{'transformers':fake}):
            result=score_local_bert(['text'],self.root,list(LABELS),batch_size=1)
            self.assertAlmostEqual(result[0]['toxic'],0.5)
            self.assertAlmostEqual(result[0]['severe_toxic'],0.7310586,places=6)
            tokenizer.assert_called_with(['text'],padding='max_length',truncation=True,max_length=256,return_tensors='pt')
            self.assertTrue(model_class.from_pretrained.call_args.kwargs['local_files_only'])
            model_class.from_pretrained.return_value=(model,{'missing_keys':['classifier.weight']})
            with self.assertRaisesRegex(ValueError,'uninitialized'):score_local_bert(['text'],self.root,list(LABELS))
            model_class.from_pretrained.return_value=(model,{})
            config.hidden_size=768
            with self.assertRaisesRegex(ValueError,'BERT-Large'):score_local_bert(['text'],self.root,list(LABELS))

    def test_reductions_are_distinct_and_explicit(self):
        scores=dict.fromkeys(LABELS,0.1);scores['threat']=0.8
        self.assertEqual(reduce_toxicity(scores,'any_label_rate',0.5),1.0)
        self.assertEqual(reduce_toxicity(0.2,'precomputed'),0.2)
        for scores,reduction,threshold in [(scores,'any_label_rate',None),(0.2,'precomputed',0.5),
                                          ({'toxic':0.2},'any_label_rate',0.5),(float('nan'),'precomputed',None),
                                          (1.1,'precomputed',None)]:
            with self.assertRaises(ValueError):reduce_toxicity(scores,reduction,threshold)

    def test_cached_scores_match_ids_and_exact_scored_text(self):
        entries=read_jsonl(self.cache);self.write(self.cache,list(reversed(entries)))
        self.assertAlmostEqual(load_cached_scores(self.cache,self.records)[0]['toxic'],0.1)
        entries[0]['text_sha256']=text_hash('another text');self.write(self.cache,entries)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):load_cached_scores(self.cache,self.records)
        self.write(self.cache,entries[1:])
        with self.assertRaisesRegex(ValueError,'IDs'):load_cached_scores(self.cache,self.records)

    def test_domain_macro_not_prompt_weighted_mean(self):
        rows,domains,categories,overall=aggregate(self.records,[1,0,0,0,0,1],[1,0,0,0,0,1],list(DOMAINS.values()))
        self.assertAlmostEqual(overall['avg_sentiment'],0.2)
        self.assertAlmostEqual(overall['avg_toxicity'],0.2)
        self.assertAlmostEqual(rows.sentiment.mean(),1/3)
        self.assertEqual(len(domains),5)
        with self.assertRaisesRegex(ValueError,'partial'):aggregate(self.records[:2],[0,0],[0,0],list(DOMAINS.values()))

    def test_missing_toxicity_model_cannot_fall_back(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):evaluate(self.normalized,self.root/'out','precomputed',{'source':'test','model':'test'})
        self.assertFalse((self.root/'out').exists())

    @unittest.skipUnless(HAS_VADER,'Install the bold extra for upstream VADER integration')
    def test_real_vader_and_cached_toxicity_end_to_end(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        out=self.root/'out'
        result=evaluate(self.normalized,out,'any_label_rate',{'source':'synthetic test only','model':'fixture'},toxicity_scores_path=self.cache,threshold=0.5)
        analyzer=SentimentIntensityAnalyzer()
        expected=[analyzer.polarity_scores(r['text'])['compound'] for r in self.records]
        self.assertAlmostEqual(result['avg_sentiment'],sum(expected)/6)
        self.assertAlmostEqual(result['avg_toxicity'],(0.5+0+0+0+1)/5)
        self.assertEqual(len(pd.read_csv(out/'per_response.csv')),6)
        manifest=json.loads((out/'provenance.json').read_text())
        self.assertEqual(manifest['sentiment']['version'],'3.3.2')
        self.assertFalse(manifest['historical_table14_reproduced'])
        self.assertEqual(load_cached_scores(out/'toxicity_scores.jsonl',self.records),load_cached_scores(self.cache,self.records))
        self.write(self.cache,[{'custom_id':r['custom_id'],'text_sha256':r['text_sha256'],'toxicity':0.2} for r in self.records])
        cached=evaluate(self.normalized,self.root/'scalar','precomputed',{'source':'synthetic scalar','model':'fixture'},toxicity_scores_path=self.cache)
        self.assertAlmostEqual(cached['avg_toxicity'],0.2)
        with self.assertRaises(FileExistsError):evaluate(self.normalized,out,'any_label_rate',{'source':'test','model':'test'},toxicity_scores_path=self.cache,threshold=0.5)

    @unittest.skipUnless(HAS_VADER,'Install the bold extra for upstream VADER integration')
    def test_upstream_vader_published_example(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        # cjhutto/vaderSentiment README's documented compound score.
        self.assertEqual(SentimentIntensityAnalyzer().polarity_scores('VADER is smart, handsome, and funny.')['compound'],0.8316)

    def test_local_checkpoint_arguments_and_architecture_rejected(self):
        # No downloads or random classifier may substitute for a supplied model.
        with self.assertRaises(ValueError):score_local_bert(['text'],str(self.root/'absent'),list(LABELS))


class PaperScopeTests(unittest.TestCase):
    def test_registry_covers_exactly_published_experiments(self):
        from llmbias.registry import functions
        from llmbias.workflows import WORKFLOWS,TASK_COVERAGE
        expected={'CAMS','SAD','medbullets','medical_bias','movielens','djinni','education_ranking','mt_gender','ecthr','ontonotes','bold','bbq'}
        self.assertEqual(set(functions),expected)
        self.assertEqual(set(TASK_COVERAGE),expected)
        self.assertIn('evaluate-bold',WORKFLOWS)
        for command in ['prepare-biasmd','prepare-disease-buster','prepare-mental-joint-labels',
                        'prepare-mental-single-label','prepare-admission-sample','prepare-admission-fields']:
            self.assertNotIn(command,WORKFLOWS)

    def test_excluded_tasks_rejected_before_reading_data(self):
        from llmbias.cli import build
        for task in ['dreaddit','education_ga','diasafety','education_ranking_generation']:
            with self.assertRaisesRegex(ValueError,'outside the paper scope'):
                build(task,'model','unused','nonexistent')


if __name__ == '__main__':
    unittest.main()
