import csv
import os
import itertools

def quora_dataset(filename):
    with open(filename, 'r') as f:
        question_tsv = csv.reader(f, delimiter = '\t')
        for row in question_tsv:
            yield int(row[0]), row[1], row[2], int(row[3])


def main():
    k = 10 # generate 1 gold and 9 distract sentences
    dataset_dir = '/mnt/disks/main/question_duplicate/dataset/raw'
    train_file = os.path.join(dataset_dir, 'train.tsv')
    dev_file = os.path.join(dataset_dir, 'dev.tsv')
    test_file = os.path.join(dataset_dir, 'test.tsv')
    train_output = os.path.join(dataset_dir, 'train_kway.tsv')
    dev_output = os.path.join(dataset_dir, 'dev_kway.tsv')
    test_output = os.path.join(dataset_dir, 'test_kway.tsv')
    questions_output = os.path.join(dataset_dir, 'questions_kway.tsv')
    dev_count = 5000
    test_count = 5000

    question_to_id = {}

    def add_to_dict(text):
        if text in question_to_id:
            return question_to_id[text]
        else:
            new_id = len(question_to_id)
            question_to_id[text] = new_id
            return new_id

    pairs = []

    for is_duplicate, q1_text, q2_text, _ in itertools.chain.from_iterable([
        quora_dataset(train_file),
        quora_dataset(dev_file),
        quora_dataset(test_file)]):

        q1_id = add_to_dict(q1_text)
        q2_id = add_to_dict(q2_text)

        if is_duplicate:
            pairs.append((q1_id, q2_id))

    print("Retrieved {} datapoints".format(len(pairs)))

    # write all the questions
    with open(questions_output, 'w+') as f:
        for question_text, q_id in question_to_id.items():
            f.write("{}\t{}\n".format(q_id, question_text))

    def write_dataset(output_file, data):
        with open(output_file, 'w+') as f:
            for q1_id, q2_id in data:
                f.write("{}\t{}\n".format(q1_id, q2_id))

    dev_split = pairs[:dev_count]
    test_split = pairs[dev_count:dev_count + test_count]
    train_split = pairs[dev_count+test_count:]

    print("Train size: {}".format(len(train_split)))
    print("Dev size: {}".format(len(dev_split)))
    print("Test size: {}".format(len(test_split)))

    write_dataset(train_output, train_split)
    write_dataset(dev_output, dev_split)
    write_dataset(test_output, test_split)


if __name__ == '__main__':
    main()
