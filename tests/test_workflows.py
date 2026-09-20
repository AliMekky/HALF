"""Synthetic offline regression checks for the paper workflows.

The oracle is selected original cell source, executed only on temporary fixtures.
No API, download, or historical output directory is used.
"""
import ast
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELLS = json.loads((Path(__file__).parent / 'fixtures/workflow_sources.json').read_text())


def tree(cell):
    return ast.parse(CELLS[str(cell)])


def execute(body, namespace):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(body), type_ignores=[])), '<original-cell>', 'exec'), namespace)
    return namespace


def definitions(cell):
    selected=[]
    for n in tree(cell).body:
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)):
            selected.append(n)
        elif isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id in {
            'LABELS','LABEL_TO_INDEX','SYSTEM_PROMPT_MEDICAL','SYSTEM_PROMPT','USER_PROMPT',
            'SYSTEM_PROMPT_TMPL','USER_PROMPT_TMPL','LANG_TAG','K','GENDERS','AGES','ETHNICITIES'} for t in n.targets):
            selected.append(n)
    return execute(selected, {'__name__':'oracle','pd':pd,'np':np})


def envelope(provider, cid, content):
    if provider=='anthropic':
        return {'custom_id':cid,'result':{'type':'succeeded','message':{'content':[{'type':'text','text':content}]}}}
    if provider=='deepseek':
        return {'custom_id':cid,'response':content}
    return {'custom_id':cid,'response':{'body':{'choices':[{'message':{'content':content}}]}}}


def jsonl(path, records):
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))


@contextlib.contextmanager
def cwd(path):
    old=Path.cwd()
    os.chdir(path)
    try:yield
    finally:os.chdir(old)


class NotebookWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_task_parsers_match_original_csv_bytes(self):
        for task, cells in [('cams',[328,330,332]),('sad',[349,351,356])]:
            for provider, cell in zip(['openai','anthropic','deepseek'],cells):
                for mode in (['onehot','indices'] if task=='sad' else [None]):
                    with self.subTest(task=task,provider=provider,mode=mode):
                        model='gpt-4.1-2025-04-14'
                        cid=f'request-0-{model}-{task.upper()}-male-adult-arab'
                        answer='OUTPUT: 2' if task=='cams' else 'Financial_Problem: 1; Other: 0; School: 1'
                        inp=self.root/f'{provider}.jsonl'
                        jsonl(inp,[envelope(provider,cid,answer),envelope(provider,cid.replace('0-','1-',1),'No answer'), {'custom_id':cid}])
                        a,b=self.root/'a.csv',self.root/'b.csv'
                        before=definitions(cell)['process_file']
                        after=importlib.import_module(f'llmbias.processing.parsing.{task}_{provider}').process_file
                        extra=[mode] if mode else []
                        with contextlib.redirect_stdout(io.StringIO()):
                            before(str(inp),str(a),*extra);after(str(inp),str(b),*extra)
                        self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_recruitment_parsers(self):
        for provider,cell in [('openai',302),('anthropic',300)]:
            inp=self.root/f'{provider}.jsonl'
            jsonl(inp,[envelope(provider,'request-0-gpt-4.1-2025-04-14-djinni-male-arab','Admit'),
                       envelope(provider,'request-1-gpt-4.1-2025-04-14-djinni-female-asian','Reject')])
            a,b=self.root/'a.csv',self.root/'b.csv'
            before=definitions(cell);before.update(INPUT_FILE=inp,OUTPUT_FILE=a)
            with contextlib.redirect_stdout(io.StringIO()):
                if provider=='anthropic':before['main']()
                else:
                    runtime=tree(cell).body
                    start=next(i for i,n in enumerate(runtime) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='records')
                    execute(runtime[start:],before)
                importlib.import_module('llmbias.processing.parsing.recruitment_'+provider).process_file(str(inp),str(b))
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_education_parser_all_providers(self):
        from llmbias.processing.parsing.education import process_jsonl
        before=definitions(364)['process_jsonl']
        for provider in ['openai','anthropic','deepseek']:
            filename='claude' if provider=='anthropic' else provider
            inp=self.root/(filename+'.jsonl')
            jsonl(inp,[envelope(provider,'request-0-gpt-4.1-2025-04-14-education_ranking-teacher-male-arab','B')])
            a,b=self.root/'a.csv',self.root/'b.csv'
            with contextlib.redirect_stdout(io.StringIO()):
                before(str(inp),str(a));process_jsonl(str(inp),str(b))
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_medical_neutral_prompt_and_parameters(self):
        from llmbias.processing.requests.medical_bias_neutral import create_batch
        inp=self.root/'gold.jsonl';jsonl(inp,[{'question':'Q?','options':{'A':'one','B':'two'}}])
        a,b=self.root/'a.jsonl',self.root/'b.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):
            definitions(265)['create_batch'](str(inp),str(a));create_batch(str(inp),str(b))
        self.assertEqual(a.read_bytes(),b.read_bytes())
        self.assertEqual(json.loads(b.read_text())['body']['temperature'],0.0)

    def test_medical_answer_cleanup(self):
        from llmbias.processing.parsing.medical_answer import extract_final_choice
        before=definitions(297)['extract_final_choice']
        for text in ['The answer is C','A\nexplanation','reasoning\nB','refused',None]:
            self.assertEqual(before(text),extract_final_choice(text))

    def test_translation_scoring_including_missing_predictions(self):
        from llmbias.evaluation.translation.scoring import evaluate
        gold=self.root/'gold.csv'
        pd.DataFrame({'gender':['male','female','male'],'type':['pro','anti','anti']}).to_csv(gold,index=False)
        inp=self.root/'judge.jsonl'
        jsonl(inp,[envelope('openai','request-0-judge-mt_gender-Arabic','male'),envelope('openai','request-1-judge-mt_gender-Arabic','male')])
        before=definitions(456);before.update(input_path=str(inp))
        loop=next(n for n in tree(456).body if isinstance(n,ast.For))
        body=copy.deepcopy(loop.body[1:-1])
        for i,n in enumerate(body):
            if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='df':
                body[i]=ast.parse(f'df = pd.read_csv({str(gold)!r})').body[0]
        expected=execute(body,before)['lang_results']
        actual=evaluate(str(inp),str(gold),str(self.root/'scores.csv'))
        pd.testing.assert_frame_equal(expected,actual)
        self.assertEqual(actual.loc['Arabic','Acc'],0.333)

    def test_translation_judge_settings(self):
        from llmbias.processing.requests.translation_judge import create_batch
        gold=self.root/'gold.csv'
        pd.DataFrame({'sentence':['The doctor arrived.'],'src_word_index':[1]}).to_csv(gold,index=False)
        inp=self.root/'translations.jsonl'
        jsonl(inp,[envelope('openai','request-0-gpt-4o-mt_gender-Arabic-pro','translated')])
        out=self.root/'requests.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):create_batch(str(gold),str(inp),str(out))
        result=json.loads(out.read_text())
        self.assertEqual(result['body']['model'],'gpt-4o-mini-2024-07-18')
        self.assertEqual(result['body']['temperature'],0)
        before=definitions(451);ns=execute([tree(452).body[0]],before)
        with contextlib.redirect_stdout(io.StringIO()):
            expected=ns['create_mt_gender_batch'](str(gold),str(inp),'gpt-4o-mini-2024-07-18',ns['base_tmpl'])
        self.assertEqual([result],expected)

    def test_recommendation_full_driver(self):
        from llmbias.evaluation.recommendation.scoring import evaluate
        data=self.root/'inputs';data.mkdir()
        for provider,filename in [('openai','gpt-4.1-2025-04-14'),('anthropic','claude'),('deepseek','deepseek_chat')]:
            for neutral in [True,False]:
                suffix='neutral' if neutral else 'male-adult-arab'
                recs=[envelope(provider,f'request-{i}-{filename}-movielens-recent-{suffix}',
                               '\n'.join(f'{j+1}. Movie {i}-{j}, 2000, genre' for j in range(10))) for i in range(2)]
                jsonl(data/f'{filename}_movielens{"_neutral" if neutral else ""}.jsonl',recs)
        ns=definitions(382)
        from collections import defaultdict
        ns.update(json=json,pd=pd,defaultdict=defaultdict,DATA_DIR=data,GENDERS={'male','female'},AGES={'minor','adult','senior'},ETHNICITIES={'arab','western','asian','african','latino'})
        expected=execute(tree(382).body[9:],ns)['df']
        with contextlib.redirect_stdout(io.StringIO()):actual=evaluate(str(data),str(self.root/'new'))
        pd.testing.assert_frame_equal(expected,actual)
        for name in ['bias_gender_pt2.csv','bias_ethnicity_pt2.csv','bias_age_pt2.csv']:
            self.assertEqual((data/name).read_bytes(),(self.root/'new'/name).read_bytes())

    def test_summarization_conversion_all_providers(self):
        from llmbias.processing.parsing.summarization import process_file
        dataset=self.root/'source.jsonl'
        jsonl(dataset,[{'article_id':'wsj_1','pair_id':1,'sample_id':0,'text':'Source','instructions':{}}])
        for provider,filename in [('openai','gpt'),('anthropic','claude'),('deepseek','deepseek')]:
            inp=self.root/(filename+'.jsonl');jsonl(inp,[envelope(provider,'request-0-model-ontonotes-wsj_1-1','Summary')])
            a,b=self.root/'a.jsonl',self.root/'b.jsonl';ns=definitions(412)
            with contextlib.redirect_stdout(io.StringIO()):
                ns['convert'](inp,ns['load_dataset'](dataset),a);process_file(str(inp),str(dataset),str(b))
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_bbq_conversion_and_scorer(self):
        from llmbias.processing.parsing.bbq import process_file
        from llmbias.evaluation.conversational.bbq import evaluate
        gold=self.root/'bbq.csv';rows=[]
        for i,polarity in enumerate(['neg','nonneg']):
            rows.append({'example_id':i,'question_index':i,'question_polarity':polarity,'context_condition':'ambig','category':'Age',
                         'answer_info':str({'ans0':['A','old'],'ans1':['B','young'],'ans2':['C','unknown']}),
                         'additional_metadata':'{}','context':'context','question':'Q?','ans0':'A','ans1':'B','ans2':'C','label':2})
        pd.DataFrame(rows).to_csv(gold,index=False)
        inp=self.root/'gpt_bbq.jsonl';jsonl(inp,[envelope('openai',f'request-{i}-model-bbq-Age-{i}-{i}','A') for i in range(2)])
        outdir=self.root/'formatted';outdir.mkdir();out=outdir/'responses.jsonl'
        process_file(str(inp),str(gold),str(out))
        ns=definitions(420);ns.update(path=str(inp),out_path=str(self.root/'oracle.jsonl'))
        runtime=[copy.deepcopy(n) for n in tree(420).body[7:] if not isinstance(n,ast.FunctionDef)]
        runtime[0]=ast.parse(f'df=pd.read_csv({str(gold)!r})').body[0]
        execute(runtime,ns)
        self.assertEqual(out.read_bytes(),(self.root/'oracle.jsonl').read_bytes())
        metadata=self.root/'metadata.csv'
        pd.DataFrame({'example_id':[0,1],'question_index':[0,1],'category':['Age','Age'],'target_loc':[0,0]}).to_csv(metadata,index=False)
        src=ast.parse((Path(__file__).parent/'fixtures/bbq_scorer.py.txt').read_text())
        body=[n for i,n in enumerate(src.body) if i not in [4,5,6,7]]
        oracle_dir=self.root/'oracle';oracle_dir.mkdir()
        with cwd(oracle_dir):
            ns=execute(body,{'result_dir':str(outdir),'metadata_file':str(metadata),'model_key':'gpt-4.1-2025-04-14'})
        with contextlib.redirect_stdout(io.StringIO()):actual=evaluate(str(outdir),str(metadata),str(self.root/'scores'),'gpt-4.1-2025-04-14')
        pd.testing.assert_frame_equal(ns['bias_df'],actual)

    def test_medical_bias_matching_and_neutral(self):
        from llmbias.evaluation.medical.bias import evaluate_prompt_matching,evaluate_neutral
        batches=self.root/'batches.jsonl';gold=self.root/'gold.jsonl';pred=self.root/'pred.csv'
        jsonl(batches,[{'custom_id':'request-0-model-medical_bias-frequency','body':{'messages':[{}, {'content':'Question one?'}]}}])
        jsonl(gold,[{'question':'Question one?','answer_idx':'A'}]);pd.DataFrame({'answer':['A']}).to_csv(pred,index=False)
        ns=definitions(442);ns.update(batches=[json.loads(batches.read_text())],responses=pd.read_csv(pred),gt=[json.loads(gold.read_text())])
        expected=execute(tree(437).body+[tree(442).body[-1]]+tree(443).body,ns)['scores']
        self.assertEqual(evaluate_prompt_matching(str(pred),str(batches),str(gold)),expected)
        self.assertEqual(evaluate_neutral(str(pred),str(gold)),{'accuracy':1.0,'count':1})

    def test_movielens_preparation(self):
        from llmbias.processing.preprocessing.datasets import prepare_movielens
        ratings=pd.DataFrame([{'userId':u,'movieId':m,'rating':4.0,'timestamp':m} for u in range(200) for m in range(10)])
        movies=pd.DataFrame([{'movieId':m,'title':f'Movie {m} (2000)','genres':'Comedy|Drama'} for m in range(10)])
        ratings.to_csv(self.root/'rating.csv',index=False);movies.to_csv(self.root/'movie.csv',index=False)
        (self.root/'recommendation_system').mkdir()
        with cwd(self.root):ns=execute(tree(220).body+tree(222).body+tree(226).body,{})
        actual=prepare_movielens(ratings.copy(),movies.copy())
        pd.testing.assert_frame_equal(ns['user_anchor_groups'],actual)
        self.assertEqual(len(actual),400)

    def test_bbq_preparation(self):
        from llmbias.processing.preprocessing.datasets import prepare_bbq
        df=pd.DataFrame([{'category':cat,'context_condition':context,'split_name':cat,'id':i} for cat in ['Age','SES'] for context in ['ambig','disambig'] for i in range(100)])
        body=tree(188).body[2:]
        expected=execute(body,{'df':df.copy()})['sampled_df']
        actual=prepare_bbq(df.copy())
        pd.testing.assert_frame_equal(expected,actual)
        self.assertEqual(len(actual),182)

    def test_recruitment_preparation(self):
        from llmbias.processing.preprocessing.datasets import match_recruitment
        cv=pd.DataFrame({'CV':['a','b'],'Primary Keyword':['Python','Python'],'Experience Years':[2,3],'English Level':['B2','B2'],'Position':['dev','dev']})
        jobs=pd.DataFrame({'Long Description':['work'],'Primary Keyword':['Python'],'Exp Years':['2y'],'English Level':['B2'],'Position':['developer']})
        random.seed(42)
        expected=execute(tree(153).body,{'df_cv':cv.copy(),'df_jobs':jobs.copy(),'pd':pd,'random':random})
        random.seed(42)
        actual=match_recruitment(cv.copy(),jobs.copy())
        pd.testing.assert_frame_equal(pd.DataFrame(expected['sampled_pairs']),actual)

    def test_legal_preprocessing_requests(self):
        inp=self.root/'legal.csv';pd.DataFrame({'text':['Case'],'applicant_gender':[0]}).to_csv(inp,index=False)
        for module,cell in [('legal_gender',386),('legal_placeholders',395)]:
            a,b=self.root/'a.jsonl',self.root/'b.jsonl'
            with contextlib.redirect_stdout(io.StringIO()):
                definitions(cell)['create_batch'](str(inp),str(a))
                importlib.import_module('llmbias.processing.requests.'+module).create_batch(str(inp),str(b))
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_imports_do_not_read_external_translation_data(self):
        # This used to fail at import without LLMBIAS_WORKSPACE.
        with patch.dict(os.environ,{},clear=True):
            importlib.import_module('llmbias.processing.requests.translation')

    def test_no_bundle_needed_for_coverage_and_help(self):
        from llmbias._workflows import TASK_COVERAGE,WORKFLOWS
        self.assertIn('evaluate-bold', TASK_COVERAGE['bold']['evaluation'])
        self.assertNotIn('education_ga', TASK_COVERAGE)
        self.assertNotIn('dreaddit', TASK_COVERAGE)
        self.assertIn('evaluate-recommendation',WORKFLOWS)
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',LLMBIAS_LEGACY_ROOT=str(self.root/'absent'))
        proc=subprocess.run([sys.executable,'-m','llmbias.cli','coverage'],cwd=self.root,env=env,text=True,capture_output=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(json.loads(proc.stdout),TASK_COVERAGE)

    def test_dispatch_and_output_guard(self):
        from llmbias._workflows import run
        inp=self.root/'answers.csv';pd.DataFrame({'answer':['The answer is A']}).to_csv(inp,index=False)
        out=self.root/'new'/'answers.csv'
        run('parse-medical-answer',{'input_path':str(inp),'output_csv_path':str(out)})
        self.assertEqual(pd.read_csv(out)['clean_answer'].tolist(),['A'])
        with self.assertRaises(FileExistsError):run('parse-medical-answer',{'input_path':str(inp),'output_csv_path':str(out)})

    def test_model_slash_only_changes_filename(self):
        from llmbias.cli import build
        data=self.root/'medical_data';data.mkdir()
        pd.DataFrame({'bias_type':['frequency'],'prompt':['Q?']}).to_csv(data/'medical_bias.csv',index=False)
        with contextlib.redirect_stdout(io.StringIO()):build('medical_data/medical_bias','org/model',self.root/'out',self.root)
        entry=json.loads((self.root/'out/org%2Fmodel_medical_bias.jsonl').read_text())
        self.assertEqual(entry['body']['model'],'org/model')

    def test_summarization_adapter_command_paths(self):
        from llmbias.evaluation.summarization.scoring import commands
        root,calls=commands('input.jsonl',str(self.root/'output'),upstream_root=self.root/'upstream',python='research-python')
        self.assertEqual(calls[0][:3],['research-python','-m','summarybias.evaluate.parse'])
        self.assertEqual(calls[1][1],str(root/'scripts/evaluate_gender_file.py'))


    def test_medical_fixed_blocks_preserve_original_order(self):
        from llmbias.evaluation.medical.bias import evaluate_blocks
        answers = ['A'] * 1273
        predictions = pd.DataFrame({'answer': answers * 6 + ['B'] * 1273})
        gold = pd.DataFrame({'answer_idx': answers})
        predictions.to_csv(self.root/'pred.csv', index=False)
        gold.to_json(self.root/'gold.jsonl', orient='records', lines=True)
        biases = [node for node in tree(275).body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'biases' for t in node.targets)]
        ns = execute(biases + tree(276).body + tree(278).body + tree(280).body,
                     {'all_prediction': predictions.copy(), 'df_jsonl': gold.copy()})
        with contextlib.redirect_stdout(io.StringIO()):
            actual = evaluate_blocks(self.root/'pred.csv', self.root/'gold.jsonl')
        self.assertEqual(list(actual), ns['biases'])
        self.assertEqual(list(actual.values()), [1.0] * 6 + [0.0])

    def test_preparation_annotations_preserve_original_cells(self):
        from llmbias.processing.preprocessing.datasets import prepare_medbullets
        df = pd.DataFrame({'question':['A 30 year old woman', 'A 12 year old boy', 'No age given', 'A 76 year old female']})
        expected = execute(tree(31).body + tree(32).body + tree(37).body, {'df':df.copy(), 're':__import__('re')})['df']
        pd.testing.assert_frame_equal(prepare_medbullets(df.copy()), expected)


if __name__=='__main__':unittest.main()
