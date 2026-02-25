import gzip
import os
import fnmatch


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


def map_path(input_path, filename_pattern_list=["*"]):
    file_map = set()
    if os.path.isfile(input_path):
        file_map.add(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for filename_pattern in filename_pattern_list:
                for filename in fnmatch.filter(files, filename_pattern):
                    full_input = os.path.join(root, filename)
                    file_map.add(full_input)
    else:
        print(
            "WARN Path is not a directory or normal file",
            'Path "{}" ignored: not a directory or normal file'.format(input_path),
        )
    return file_map
