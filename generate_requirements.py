import ast
import sys

def extract_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    return sorted(imports)

def write_requirements(imports, output_file):
    with open(output_file, 'w') as file:
        for module in imports:
            file.write(module + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_requirements.py <python_file> <output_requirements_file>")
        sys.exit(1)

    python_file = sys.argv[1]
    output_requirements_file = sys.argv[2]

    imports = extract_imports(python_file)
    write_requirements(imports, output_requirements_file)

    print(f"Requirements written to {output_requirements_file}")