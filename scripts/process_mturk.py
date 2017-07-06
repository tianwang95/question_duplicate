import csv
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action = 'store', dest = 'input_file')
    parser.add_argument('--output', action = 'store', dest = 'output_file')

    options = parser.parse_args()

    ## Print used options
    for arg in vars(options):
        print("{}\t{}".format(arg, getattr(options, arg)))

    return options

def main():
    options = parse()

if __name__ == "__main__":
    main()
