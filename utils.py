import pandas as pd
import numpy as np
import argparse

def parse_filename():
    """
    Parses path/to/data.arff from command line.
    """
    parser = argparse.ArgumentParser(description="kMeans clustering algorithm",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("filename", type=str, help="path to .arff file")

    parser.add_argument("-N", metavar="\b", type=int, default=100,
                        help="number of iterations")
    args = vars(parser.parse_args())
    return args

def readArff(filename):
    """
    Same readArff function with an update for real values
    """
    with open (filename, 'r') as f:
        # split lines, remove ones with comments
        lines = [line.lower() for line in f.read().split('\n') if not line.startswith('%')]

    # remove empty lines
    lines = [line for line in lines if line != '']

    columns = []
    data = []
    for index, line in enumerate(lines):
        if line.startswith('@attribute'):
            columns.append(line)

        if line.startswith('@data'):
            # get the rest of the lines excluding the one that says @data
            data = lines[index+1:]
            break

    # clean column names -- '@attribute colname  \t\t\t{a, b, ...}'
    cleaned_columns = [c[11:c.index('real')].strip() for c in columns[:-1]]

    # ** change for real values. skip last column and parse differently
    class_val = columns[-1]
    cleaned_columns.append(class_val[11:class_val.index('{')].strip())

    # clean and split data
    cleaned_data = [d.replace(', ', ',').split(',') for d in data]

    # create dataframe
    return pd.DataFrame(cleaned_data, columns = cleaned_columns)


def preprocess_data(df):
    """
    Split into X and y and return values as numpy arrays
    """
    ys = df.iloc[:,-1]
    ys = ys.values

    # change xs to 2d numpy array -- convert strings to floats
    xs = df.iloc[:,:-1].astype(float)
    xs = xs.values

    return xs, ys

def euclidean_distance(x1, x2):
    """ Calculates the euclidean distance between two points """
    assert np.size(x1) == np.size(x2)

    # Squared distance between each coordinate
    distances = np.square(x1 - x2)
    return np.sqrt(sum(distances))

def inaccuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the % inaccuracy and counts """
    pct = round(100 * (np.sum(y_true != y_pred, axis=0) / len(y_true)), 2)
    number = np.sum(y_true != y_pred, axis=0)
    return pct, number
