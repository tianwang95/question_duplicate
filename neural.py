import csv
import numpy as np
from data import Data

def main():
    # glove = glove2dict('dataset/glove.6B.50d.txt')
    data = Data('dataset/raw/quora_duplicate_questions.tsv')
    for x, y, z in data.generator():
        print(x)
        print(y)
        print(z)

if __name__ == '__main__':
    main()

