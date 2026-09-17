"""Rebuild notebook extractions without executing the notebook (stdlib only).

Generated modules contain ordinary Python functions, not a notebook interpreter.
Selections deliberately preserve historical algorithms, including known limitations.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / 'legacy/data_analysis.ipynb').read_text())
PROVENANCE = []
USED = set()


def nodes(cell):
    USED.add(cell)
    return ast.parse(''.join(NB['cells'][cell]['source'])).body


def definitions(cell, constants=()):
    return [n for n in nodes(cell) if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]


def parse(text):
    return ast.parse(text).body


def function(signature, body, before='', after=''):
    fn = ast.parse('def ' + signature + ':\n    pass').body[0]
    fn.body = parse(before) + copy.deepcopy(body) + parse(after)
    return ast.fix_missing_locations(fn)


def write(module, cells, body):
    path = ROOT / 'src/llmbias' / (module + '.py')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f'"""Historical notebook cells {cells}; structural extraction, unchanged scientific logic."""\n'
    text += ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))) + '\n'
    ast.parse(text)
    path.write_text(text)
    PROVENANCE.append({'module': 'llmbias.' + module.replace('/', '.'), 'cells': cells,
                       'operation': 'definitions extracted; top-level processing wrapped; file paths supplied as arguments'})


for cell, name in [(328, 'cams_openai'), (330, 'cams_anthropic'), (332, 'cams_deepseek'),
                   (349, 'sad_openai'), (351, 'sad_anthropic'), (356, 'sad_deepseek'), (364, 'education')]:
    write('parsing/' + name, [cell], definitions(cell, ('LABELS', 'LABEL_TO_INDEX')))

body = definitions(300)
main = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == 'main')
body.remove(main)
body.append(function('process_file(input_path, output_csv_path)', main.body,
                     'INPUT_FILE = Path(input_path)\nOUTPUT_FILE = Path(output_csv_path)'))
write('parsing/recruitment_anthropic', [300], body)
write('parsing/recruitment_openai', [302], definitions(302) + [function(
    'process_file(input_path, output_csv_path)', nodes(302)[9:],
    'INPUT_FILE = Path(input_path)\nOUTPUT_FILE = Path(output_csv_path)')])
# Locate the runtime block by its variable, avoiding dependence on comment line counts.
runtime = nodes(302)
start = next(i for i,n in enumerate(runtime) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'records')
write('parsing/recruitment_openai', [302], definitions(302) + [function(
    'process_file(input_path, output_csv_path)', runtime[start:],
    'INPUT_FILE = Path(input_path)\nOUTPUT_FILE = Path(output_csv_path)')])
write('parsing/medical_answer', [297], definitions(297) + parse('''
def process_file(input_path, output_csv_path):
    df = pd.read_csv(input_path)
    df['clean_answer'] = df['answer'].apply(extract_final_choice)
    df.to_csv(output_csv_path, index=False)
'''))
write('preparation/medical_bias_neutral', [265], definitions(265, ('SYSTEM_PROMPT_MEDICAL',)))
write('preparation/legal_gender', [386], definitions(386, ('SYSTEM_PROMPT',)))
write('preparation/legal_placeholders', [395], definitions(395, ('SYSTEM_PROMPT', 'USER_PROMPT')))
write('parsing/summarization', [412], definitions(412) + parse('''
def process_file(input_path, dataset_path, output_path):
    convert(pathlib.Path(input_path), load_dataset(pathlib.Path(dataset_path)), pathlib.Path(output_path))
'''))

# A separate historical version, not a replacement for the existing cell 451 module.
write('analysis/translation_judge_cell450', [450], definitions(450, ('SYSTEM_PROMPT_TMPL','USER_PROMPT_TMPL','LANG_MAP')))
write('analysis/cams_intersectional', [341], definitions(341))
write('analysis/recruitment_cell304', [304], definitions(304))

# Translation scoring: source cell 456 is a per-model loop; remove only the paths
# and final export, making a single-model function with explicit paths.
loop = next(n for n in nodes(456) if isinstance(n, ast.For))
body = copy.deepcopy(loop.body[1:-1])
for i,n in enumerate(body):
    if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'df':
        body[i] = parse('df = pd.read_csv(gold_csv)')[0]
write('evaluation/translation', [456], parse('import json\nimport pandas as pd') +
      [n for n in definitions(456) if isinstance(n,ast.FunctionDef)] +
      [function('evaluate(input_path, gold_csv, output_path)', body, after='lang_results.to_csv(output_path)\nreturn lang_results')])

# Preserve the cell 452 judge configuration (not the translation generation model).
write('preparation/translation_judge', [451,452], parse('''
import copy
import json
from llmbias.analysis.translation_judge import create_mt_gender_batch
''') + [nodes(452)[0]] + parse('''
JUDGE_MODEL = 'gpt-4o-mini-2024-07-18'
def create_batch(gold_csv, translations_jsonl, output_path, model_name=JUDGE_MODEL):
    batch = create_mt_gender_batch(gold_csv, translations_jsonl, model_name, copy.deepcopy(base_tmpl))
    with open(output_path, 'w', encoding='utf-8') as f:
        for req in batch:
            f.write(json.dumps(req, ensure_ascii=False) + '\\n')
'''))

runtime=copy.deepcopy(nodes(382)[9:])
for i,n in enumerate(runtime):
    if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='outdir':
        runtime[i]=parse('outdir = Path(output_dir)')[0]
write('evaluation/recommendation', [382], parse('''
from pathlib import Path
from collections import defaultdict
import pandas as pd
from llmbias.analysis.recommendation import load_jsonl, req_idx_of, query_type_of, group_of, extract_items, split_attrs, jaccard, prag
''') + [function('evaluate(data_dir, output_dir)', runtime,
    'DATA_DIR = Path(data_dir)\nPath(output_dir).mkdir(parents=True, exist_ok=True)', 'return df')])

# Earlier recommendation version retains the original grouping, denominator,
# and first-neutral selection; expose it explicitly, never as the default.
old=nodes(379)
old_defs=definitions(379,('K','GENDERS','AGES','ETHNICITIES'))
start=next(i for i,n in enumerate(old) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='records')
old_runtime=copy.deepcopy(old[start:])
# Notebook globals().setdefault was state storage, replace only its namespace with
# an explicit local loaded dictionary. The loop and scoring remain unchanged.
for n in ast.walk(ast.Module(body=old_runtime,type_ignores=[])):
    if isinstance(n,ast.For):
        n.body=[x for x in n.body if not (isinstance(x,ast.Expr) and ast.unparse(x).startswith("globals().setdefault"))]
write('evaluation/recommendation_cell379', [379], old_defs + [function(
    'evaluate(data_dir)',old_runtime,'DATA_DIR = Path(data_dir)\nloaded = {}',
    "return {'gender': gender_summary, 'ethnicity': eth_summary}")])

# BBQ conversion preserves its historical model-key column and provider detection.
bbq=nodes(420)
body=[copy.deepcopy(n) for n in bbq[7:] if not isinstance(n,ast.FunctionDef)]
body[0]=parse('df = pd.read_csv(gold_csv)')[0]
write('parsing/bbq', [420], definitions(420) + [function(
    'process_file(input_path, gold_csv, output_path)',body,
    'path = str(input_path)\nout_path = str(output_path)')])

# Copy the active local upstream scorer as a callable, preserving its formulas.
upstream=ROOT/'legacy/experiments/conv_ai/evaluation/bbq/BBQ/analysis_scripts/BBQ_bias_score.py'
uptext=upstream.read_text();upnodes=ast.parse(uptext).body
updefs=[n for n in upnodes if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
start=next(i for i,n in enumerate(upnodes) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='metadata')
runtime=[copy.deepcopy(n) for n in upnodes[start:] if not isinstance(n,ast.FunctionDef)]
for n in ast.walk(ast.Module(body=runtime,type_ignores=[])):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to_csv':
        n.args[0]=ast.BinOp(left=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Name(id='output_dir',ctx=ast.Load())],keywords=[]),op=ast.Div(),right=n.args[0])
write('evaluation/bbq', [], parse('from pathlib import Path')+updefs+[function(
    "evaluate(result_dir, metadata_file, output_dir, model_key='o4-mini-2025-04-16')",runtime,
    'Path(output_dir).mkdir(parents=True, exist_ok=True)', 'return bias_df')])
PROVENANCE[-1]['source_file']=str(upstream.relative_to(ROOT))
PROVENANCE[-1]['sha256']=hashlib.sha256(uptext.encode()).hexdigest()

# Medical scoring versions: retain positional/block and prompt-matching paths.
imports=parse('import json\nimport pandas as pd\nimport re\nfrom difflib import SequenceMatcher')
functions=[n for n in definitions(442) if isinstance(n,ast.FunctionDef)]
matching=nodes(437)+[nodes(442)[-1]]+nodes(443)
positional=nodes(275)+nodes(276)+nodes(278)+nodes(280)
# Return the per-type accuracies previously printed, without changing computation.
for n in ast.walk(ast.Module(body=positional,type_ignores=[])):
    if isinstance(n,ast.For) and ast.unparse(n.target)=='bias' and any(isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='accuracy' for x in n.body):
        n.body += parse('scores[bias] = accuracy')
write('evaluation/medical_bias', [273,275,276,278,280,287,437,442,443], imports+functions+[
    function('evaluate_prompt_matching(responses_csv, batches_path, gold_path)',matching,
      "responses = pd.read_csv(responses_csv)\nwith open(batches_path) as f:\n    batches = [json.loads(line) for line in f if line.strip()]\nwith open(gold_path) as f:\n    gt = [json.loads(line) for line in f if line.strip()]",'return scores'),
    function('evaluate_blocks(predictions_csv, gold_path)',positional,
      "all_prediction = pd.read_csv(predictions_csv)\ndf_jsonl = pd.read_json(gold_path, lines=True)\nscores = {}",'return scores'),
    function('evaluate_neutral(predictions_csv, gold_path)',nodes(271)+nodes(273),
      'df_jsonl = pd.read_json(gold_path, lines=True)\noutput = pd.read_csv(predictions_csv)','return {"accuracy": accuracy, "count": total}'),
    function('evaluate_cleaned(all_prediction)',[n for n in nodes(287) if not isinstance(n,ast.Import)],after='return accuracy_per_bias')])

# Notebook utilities; no top-level I/O is retained.
write('preparation/response_files', [448,472], definitions(472)+[
    function('reformat_claude(input_path, output_path)',nodes(448)[3:])])

# Dataframe recipes: calculations remain the original statements, while source
# acquisition and output paths are the caller's responsibility.
recipe_imports=parse('import pandas as pd\nimport numpy as np\nimport random\nimport re\nimport json\nimport ast\nfrom pathlib import Path')
recipes=[]
def recipe(signature, cells, selection=None, before='', after=''):
    body=[]
    for cell in cells:
        ns=nodes(cell)
        body += ns if selection is None else selection(cell,ns)
    recipes.append(function(signature,body,before,after))

recipe('prepare_education(frames)', [], before='df = pd.concat(frames, ignore_index=True)\ndf.drop_duplicates(inplace=True)',after='return df')
USED.update([100,101,104])
recipe('prepare_bbq(df)',[188],lambda c,ns:ns[2:-3],after='return sampled_df')
# Override via explicit statement selection to exclude dataset loading and prints.
recipes[-1]=function('prepare_bbq(df)',[n for n in nodes(188) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ['df','sampled_df']][1:],after='return sampled_df')
recipe('match_recruitment(df_cv, df_jobs)',[153],after='return pd.DataFrame(sampled_pairs)')
recipe('prepare_movielens(ratings, movies)',[220,222,226],
 lambda c,ns:[n for n in ns if not isinstance(n,(ast.Import,ast.ImportFrom))
    and not (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='pd.read_csv')
    and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='to_csv')],after='return user_anchor_groups')
recipe('sample_medical_bias(merged_df)',[48],lambda c,ns:[n for n in ns if not isinstance(n,ast.Import)],after='return df_sampled')
recipe('sample_mental_multilabel(df)',[71],after="return {'sampled': df_sampled, 'remaining': df_remaining}")
recipe('prepare_translation(pro_path, anti_path)',[],before="df = pd.read_csv(pro_path, sep='\\t', header=None, names=['gender','src_word_index','sentence','profession'])\ndf2 = pd.read_csv(anti_path, sep='\\t', header=None, names=['gender','src_word_index','sentence','profession'])",after="df2['type'] = 'anti'\ndf['type'] = 'pro'\ndf = pd.concat([df, df2], ignore_index=True)\nreturn df")
USED.update([111,114,115,117])
recipe('attach_neutralized_text(df, responses_path)',[322,323],before="with open(responses_path) as f:\n    data_list = [json.loads(line) for line in f if line.strip()]",after='return df')
recipe('update_legal_gender(csv_path, jsonl_path)',[387],lambda c,ns:ns[4:6],after='return df')
recipe('prepare_ontonotes(input_path)',[403],lambda c,ns:ns[2:5],before='jsonl_path = Path(input_path)',after='return df')
recipe('prepare_medbullets(df)',[31,32,37],lambda c,ns:[n for n in ns if not isinstance(n,ast.Import)],after='return df')
recipe('sample_medbullets_by_gender(df)',[40],after='return df_sampled')
recipe('prepare_medical_prompts(base_dir)',[213,216],lambda c,ns:[n for n in ns if not isinstance(n,(ast.Import,ast.ImportFrom))],before='',after='return df')
# The collect_prompts definition needs its call before applying cell 216.
recipes[-1].body.insert(-2,parse('df = collect_prompts(base_dir)')[0])
write('preparation/notebook_recipes',sorted(USED.intersection(set(range(0,235))|{322,323,387,403})),recipe_imports+recipes)

# Reporting remains opt-in; imports of plotting libraries occur on invocation.
plot_defs=[n for n in definitions(308,('PALETTE','THRESHOLD')) if not isinstance(n,(ast.Import,ast.ImportFrom))]
plot_fn=next(n for n in plot_defs if isinstance(n,ast.FunctionDef))
plot_fn.body=parse('import matplotlib.pyplot as plt\nimport seaborn as sns')+plot_fn.body
plot_loop=copy.deepcopy(nodes(308)[-1])
for n in ast.walk(plot_loop):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_plot':
        n.args[2]=ast.Call(func=ast.Name(id='str',ctx=ast.Load()),args=[ast.BinOp(left=ast.Name(id='outdir',ctx=ast.Load()),op=ast.Div(),right=n.args[2])],keywords=[])
write('reporting/recruitment',[308],parse('import pandas as pd\nfrom pathlib import Path')+plot_defs+[
    function('report(input_path, output_dir)',[plot_loop],
     "import seaborn as sns\ndf = pd.read_csv(input_path)\nsns.set_theme(style='whitegrid')\noutdir = Path(output_dir)\noutdir.mkdir(parents=True, exist_ok=True)")])
write('reporting/legal',[418],definitions(418))

# Preserve the short qualitative enrichment cells as explicit operations too.
write('reporting/qualitative',[429,433,495,496,497],parse('import json\nimport pandas as pd')+[
    function('enrich_medbullets(outputs, batches, gt)',nodes(429),after='return outputs'),
    function('enrich_deepseek(outputs, batches)',nodes(433),after='return outputs'),
    function('enrich_flip_cases(detailed, data, gt_answers)',nodes(495)+nodes(496)+nodes(497),after='return detailed')])

# Serialize reviewed, nonsecret cell source as a small independent preservation
# oracle. These are data, never executed by package runtime.
USED.update(i for entry in PROVENANCE for i in entry['cells'])
fixture={str(i):''.join(NB['cells'][i]['source']) for i in sorted(USED)}
for text in fixture.values():
    if re.search(r'(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,})',text):
        raise ValueError('Credential-like literal in selected source fixture')
fp=ROOT/'tests/fixtures/notebook_cells.json';fp.parent.mkdir(parents=True,exist_ok=True)
fp.write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n')
(ROOT/'tests/fixtures/bbq_scorer.py.txt').write_text(uptext)
for entry in PROVENANCE:
    entry['cell_sha256']={str(i):hashlib.sha256(fixture[str(i)].encode()).hexdigest() for i in entry['cells']}
# Deduplicate provisional emissions of the same module.
PROVENANCE=list({x['module']:x for x in PROVENANCE}.values())
(ROOT/'docs/notebook-extractions.json').write_text(json.dumps(PROVENANCE,indent=2)+'\n')
print('Extracted',len(PROVENANCE),'modules; preserved',len(fixture),'source cells')
