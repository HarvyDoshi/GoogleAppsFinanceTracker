import os

def generate_tree(dir_path, prefix=''):
    tree_str = ''
    ignore_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache', 'wandb', '.idea', '.vscode'}
    try:
        items = sorted(os.listdir(dir_path))
    except Exception:
        return ''
    items = [i for i in items if i not in ignore_dirs]
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        path = os.path.join(dir_path, item)
        connector = '└── ' if is_last else '├── '
        tree_str += prefix + connector + item + '\n'
        if os.path.isdir(path):
            extension = '    ' if is_last else '│   '
            tree_str += generate_tree(path, prefix=prefix + extension)
    return tree_str

root = r'C:\Users\harvy\Downloads\trace_project\trace'
tree_output = '# Project Structure\n\n```text\ntrace/\n' + generate_tree(root) + '```\n'
with open(os.path.join(root, 'project_structure.md'), 'w', encoding='utf-8') as f:
    f.write(tree_output)
