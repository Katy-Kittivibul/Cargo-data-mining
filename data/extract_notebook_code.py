
import json
import sys

def extract_code(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                out.write(f"\n# --- CELL {i} ---\n{source}\n")
            elif cell['cell_type'] == 'markdown':
                source = ''.join(cell['source'])
                # Comment out markdown
                commented_source = '\n'.join(['# ' + line for line in source.splitlines()])
                out.write(f"\n# --- MARKDOWN {i} ---\n{commented_source}\n")

if __name__ == '__main__':
    extract_code(sys.argv[1], sys.argv[2])
