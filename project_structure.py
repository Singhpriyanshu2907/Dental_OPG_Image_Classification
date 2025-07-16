import os
import pathlib


list_of_files = [
    "github/workflows/.gitkeep",
    "src/__init__.py",
    "src/custom_exception.py",
    "src/logger.py",
    "artifacts/",  
    "config/__init__.py",  
    "notebooks/experiments.ipynb",
    "templates/",
    "utils/__init__.py",
    "requirements.txt"
]


for filepath in list_of_files:
    filepath = pathlib.Path(filepath)

    if filepath.suffix or filepath.name == '.gitkeep':
        filedir = filepath.parent

        if filedir and not filedir.exists():
            os.makedirs(filedir, exist_ok = True)

        
        if not filepath.exists():
            with open(filepath, "w") as f:
                pass
            print(f"Created empty file at: {filepath} ")
    else:
        if not filepath.exists():
            os.makedirs(filepath, exist_ok = True)
            print(f"Created directory: {filepath}")