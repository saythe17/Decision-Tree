import pandas as pd
import sys
from math import log


def main():
    # read
    dataset = pd.read_csv(str(sys.argv[1]), sep='\t', header=0)

    # get the output filename
    output = str(sys.argv[2])

    # compute the number of samples
    max_num = dataset.shape[0]

    # compute the number of each label in the last column
    last_column = dataset.columns[-1]
    num = dataset[last_column].value_counts().max()
    p1 = num / max_num
    p2 = 1 - p1

    # compute entropy and error rate
    entropy = - p1 * log(p1, 2) - p2 * log(p2, 2)
    error = p2

    # output
    fp = open(output, "w")
    print("entropy:", entropy, file=fp)
    print("error:", error, file=fp)
    fp.close()

if __name__ == "__main__":
    main()
