import os
import json

def generate_package_structure(start_path, output_file):
    """Generate a tree-like structure of the PyMDP package"""
    
    def get_structure(start_path):
        structure = {}
        for item in os.listdir(start_path):
            path = os.path.join(start_path, item)
            if os.path.isfile(path):
                if path.endswith('.py') or path.endswith('.gnn') or path.endswith('.json'):
                    structure[item] = "file"
            elif os.path.isdir(path) and not item.startswith('.') and not item == '__pycache__':
                structure[item] = get_structure(path)
        return structure

    package_structure = {
        "pymdp": get_structure(start_path)
    }
    
    # Write structure to file
    with open(output_file, 'w') as f:
        json.dump(package_structure, f, indent=4)
        
    # Also create a more readable text version
    with open(output_file.replace('.json', '.txt'), 'w') as f:
        def write_structure(structure, prefix=""):
            for key, value in sorted(structure.items()):
                if value == "file":
                    f.write(f"{prefix}├── {key}\n")
                else:
                    f.write(f"{prefix}├── {key}/\n")
                    write_structure(value, prefix + "│   ")
                    
        write_structure(package_structure)

def generate_methods_documentation(start_path, output_file):
    """Generate documentation of all methods in the package"""
    
    def get_python_files(start_path):
        python_files = []
        for root, dirs, files in os.walk(start_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    with open(output_file, 'w') as f:
        for py_file in sorted(get_python_files(start_path)):
            rel_path = os.path.relpath(py_file, start_path)
            f.write(f"\n{'='*80}\n")
            f.write(f"File: {rel_path}\n")
            f.write(f"{'='*80}\n\n")
            
            try:
                with open(py_file, 'r') as source:
                    f.write(source.read())
            except Exception as e:
                f.write(f"Error reading file: {str(e)}\n")

if __name__ == "__main__":
    # Get the PyMDP root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pymdp_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Generate structure documentation
    structure_file = os.path.join(current_dir, "pymdp_structure.json")
    generate_package_structure(pymdp_root, structure_file)
    
    # Generate methods documentation
    methods_file = os.path.join(current_dir, "pymdp_methods.txt")
    generate_methods_documentation(pymdp_root, methods_file)
    
    print(f"Package structure saved to: {structure_file}")
    print(f"Methods documentation saved to: {methods_file}")
