import sys
sys.path.append('../')
import pickle
from data_point import DataPoint
import os
import gzip
import struct


def main():
    directory = "../dataset/pickled"
    files = os.listdir(directory)
    files = sorted(files, key=lambda x : int(x.split("-")[0]))
    try:
        annotated_file = gzip.open("quora_annotated.data.gz", 'wb')
        count = 0
        for name in files:
            with open(os.path.join(directory, name), 'rb') as f:
                points = pickle.load(f)
                for point in points:
                    annotated_file.write(get_packed_msg(point.q1_proto))
                    annotated_file.write(get_packed_msg(point.q2_proto))
                    count += 1
                    if (count % 100 == 0):
                        print(count)
    finally:
        annotated_file.close()

def get_packed_msg(proto):
    size_str = struct.pack("i", len(proto))
    assert(len(size_str) == 4)
    return size_str + proto

main()