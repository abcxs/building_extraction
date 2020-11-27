import glob
import itertools
import pickle


def gen_txt(input_pkl, output_file):
    with open(input_pkl, 'rb') as f:
        polygons = pickle.load(f)

    result = []
    for polygon in polygons:
        p = ' '.join(list(map(str, itertools.chain.from_iterable(polygon[:-1]))))
        result.append(p)
    with open(output_file, 'w') as f:
        f.write('\n'.join(result))
