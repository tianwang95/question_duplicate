import csv
import argparse
import random

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-file', action='store', dest='prediction_file')
    parser.add_argument('--output-file', action='store', dest='output_file')

    options = parser.parse_args()

    ## Print used options
    for arg in vars(options):
        print("{}\t{}".format(arg, getattr(options, arg)))

    return options

def main():
    options = parse()

    head_row = []
    for i in range(5):
        head_row += ["target_question_id_{}".format(i), "target_question_text_{}".format(i)]
        for j in range(10):
            head_row += ["choice_question_id_{}_{}".format(i, j), "choice_question_text_{}_{}".format(i, j)]

    ### Load questions
    question_tsv = "../dataset/raw/questions_kway.tsv"
    questions = {}
    with open(question_tsv, 'r') as q_file:
        question_tsv = csv.reader(q_file, delimiter = '\t')
        for row in question_tsv:
            questions[int(row[0])] = row[1]

    ### Load predictions
    with open(options.prediction_file, 'r') as f:
        prediction_tsv = csv.reader(f)
        predictions = list(prediction_tsv)

    ### Write output tsv
    with open(options.output_file, 'w+') as o:
        output_csv = csv.writer(o)
        output_csv.writerow(head_row)
        for i in range(0, len(predictions), 5):
            if i % 1000 == 0:
                print(i)
            new_row = []
            for k in range(5):
                new_row += [predictions[i+k][0], questions[int(predictions[i+k][0])]] #target id and target text
                choice_ids = predictions[i+k][1:11]
                random.shuffle(choice_ids)
                for choice_id in choice_ids:
                    new_row += [choice_id, questions[int(choice_id)]] #choice id and choice text

            output_csv.writerow(new_row)

    print("Done")
if __name__ == "__main__":
    main()
