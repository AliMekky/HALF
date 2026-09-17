"""Historical notebook cells [448, 472]; structural extraction, unchanged scientific logic."""
import json

def concat_jsonl_files(input_paths, output_path):
    with open(output_path, 'w', encoding='utf-8') as fout:
        for path in input_paths:
            with open(path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith('\n') else line + '\n')

def reformat_claude(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    raw_blocks = content.split('}\n{')
    json_blocks = []
    for (i, block) in enumerate(raw_blocks):
        if i == 0:
            json_blocks.append(block + '}')
        elif i == len(raw_blocks) - 1:
            json_blocks.append('{' + block)
        else:
            json_blocks.append('{' + block + '}')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for block in json_blocks:
            try:
                obj = json.loads(block)
                json.dump(obj, out_f, ensure_ascii=False)
                out_f.write('\n')
            except json.JSONDecodeError as e:
                print('Error parsing block:\n', block[:100], '\n', e)
