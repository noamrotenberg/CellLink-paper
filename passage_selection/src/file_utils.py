import fnmatch
import gzip
import hashlib
import os
import warnings
from pathlib import Path

def read_pmids(filename):
    pmids = set()
    open_func = gzip.open if filename.endswith(".gz") else open
    with open_func(filename, "rt") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            pmids.add(line)
    return pmids

def get_filenames(path, filename_pattern_list=["*"]):
    filenames = []
    if os.path.isfile(path):
        filenames.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename_pattern in filename_pattern_list:
                for filename in fnmatch.filter(files, filename_pattern):
                    filenames.append(os.path.join(root, filename))
    else:
        warnings.warn(
            f"Path '{path}' ignored: not a directory or file", RuntimeWarning
        )
    return filenames

def build_file_map(input_path, output_path, filename_pattern_list=["*"], create_parents = True):
    """
    Build a mapping from input file paths to corresponding output file paths.

    If `input_path` is a file, maps directly to `output_path`.
    If `input_path` is a directory, walks recursively and mirrors structure.
    """
    file_map = {}
    if os.path.isfile(input_path):
        file_map[input_path] = output_path
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for filename_pattern in filename_pattern_list:
                for filename in fnmatch.filter(files, filename_pattern):
                    full_input = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_input, input_path)
                    full_output = os.path.join(output_path, rel_path)
                    file_map[full_input] = full_output
                    if create_parents:
                        Path(full_output).parent.mkdir(parents=True, exist_ok=True)
    else:
        warnings.warn(
            f"Path '{input_path}' ignored: not a directory or file", RuntimeWarning
        )
    return file_map


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as file:
        while chunk := file.read(4096):
            md5.update(chunk)
    return md5.hexdigest()
