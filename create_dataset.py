from pathlib import Path
import numpy as np
from labml import logger, lab

class PythonFile():

    def __init__(self, relative_path, path):
        self.relative_path = relative_path
        self.path = path

class GetPythonFiles:

    def __init__(self):
        self.source_path = Path(lab.get_data_path() / 'source')
        self.files = []
        self.add_all_files(self.source_path)
        logger.inspect([f.path for f in self.files])

    def add_file(self, path):
        tmp = path.relative_to(self.source_path).parents
        relative_path = path.relative_to(self.source_path / tmp[len(tmp) - 1])
        self.files.append(PythonFile(relative_path=str(relative_path), path=path))

    def add_all_files(self, path):
        for p in path.iterdir():
            if p.is_dir():
                self.add_all_files(p)
            elif p.suffix == '.py':
                self.add_file(p)

def read_file(path):
    with open(str(path)) as f:
        content = f.read()
    return content

def write_file(path, source_files):
    with open(str(path), 'w') as f:
        for source in source_files:
            f.write(read_file(source.path) + "\n")

def main():
    source_files = GetPythonFiles().files
    np.random.shuffle(source_files)
    
    train_valid_split = int(len(source_files) * 0.9)
    write_file(lab.get_data_path() / 'train.py', source_files[:train_valid_split])
    write_file(lab.get_data_path() / 'valid.py', source_files[train_valid_split:])

if __name__ == '__main__':
    main()