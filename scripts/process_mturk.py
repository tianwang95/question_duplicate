import csv
import numpy as np
import argparse
from collections import Counter
import os
from colorama import init, Fore, Style
init()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', action = 'store', dest = 'model_dir', required=True)
    parser.add_argument('--agree-percent', action='store', dest='agree_percent', required=True, type=float)
    parser.add_argument('--no-view', action='store_false', dest='is_view')

    options = parser.parse_args()

    ## Print used options
    for arg in vars(options):
        print("{}\t{}".format(arg, getattr(options, arg)))

    return options

class Turker(object):
    """
    1 Task = 1 target question
    """
    def __init__(self, turk_id):
        self.turk_id = turk_id
        self.tasks_completed = []
        self.num_judgments = 0
        self.num_agreements = 0
        self.agree_percent = 0.0

class Task(object):
    def __init__(self, turk_id, target_id, choice_questions, selected_questions):
        self.task_id = turk_id + '_' + str(target_id)
        self.turk_id = turk_id
        self.target_id = target_id
        self.choice_questions = choice_questions
        self.selected_questions = selected_questions

def main():
    options = parse()
    input_file = os.path.join(options.model_dir, "mturk_results.csv")
    output_file = os.path.join(options.model_dir, "human_eval.csv")
    choices_file = os.path.join(options.model_dir, "eval_raw.csv")

    tasks = {}
    turkers = {}
    with open(input_file, 'r') as f:
        results_csv = csv.reader(f)
        top_row = next(results_csv)
        hit_index = top_row.index("HITId")
        turk_index = top_row.index("WorkerId")
        input_start = top_row.index('Input.target_question_id_0') 
        result_start = top_row.index('Answer.question_0')
        hit_batches = {}
        for row in results_csv:
            hit_id = row[hit_index]
            new_tasks =  process_row(row, turk_index, input_start, result_start)
            ### Add to hit_batches
            if row[hit_index] not in hit_batches:
                hit_batches[hit_id] = []
            hit_batches[hit_id].append(new_tasks)
            ### Process tasks
            for task in new_tasks:
                ### Add to tasks
                tasks[task.task_id] = task
                ### Add to turkers
                if task.turk_id not in turkers:
                    turkers[task.turk_id] = Turker(task.turk_id)
                ### Associate task with Turker
                turkers[task.turk_id].tasks_completed.append(task.task_id)

    ### Count agreement by looping through batches
    for hit_id, assignments in hit_batches.items():
        ### assignments is list of list of tasks
        for task_idx in range(len(assignments[0])): ### number of tasks in one assignment
            for target_assign_idx in range(len(assignments)):
                for compare_assign_idx in range(len(assignments)):
                    if target_assign_idx != compare_assign_idx:
                        target_task = assignments[target_assign_idx][task_idx]
                        compare_task = assignments[compare_assign_idx][task_idx]
                        target_bool_arr = get_bool_array(target_task.choice_questions,
                                                         target_task.selected_questions)
                        compare_bool_arr = get_bool_array(compare_task.choice_questions,
                                                          compare_task.selected_questions)
                        curr_turker = turkers[target_task.turk_id]
                        curr_turker.num_judgments += len(target_bool_arr)
                        curr_turker.num_agreements += compute_num_agree(target_bool_arr, compare_bool_arr)

    ### compute percentages for everyone
    for turker in turkers.values():
        turker.agree_percent = float(turker.num_agreements) / turker.num_judgments

    ### compute std deviation
    percentages = np.asarray([turker.agree_percent for turker in turkers.values()])
    std = np.std(percentages)
    mean = np.mean(percentages)
    cutoff = 0.7
    print("Cutoff agree percentage:\t{:.3f}".format(cutoff))

    ### filter out turkers with crappy agreement percentages
    for turker in turkers.values():
        if turker.agree_percent < cutoff:
            print(turker.turk_id)
            for task_id in turker.tasks_completed:
                del tasks[task_id]
                print(len(tasks))

    ### compile votes for each target_id
    target_id_to_votes = {}
    num_tasks_for_target_id = Counter()
    for task in tasks.values():
        num_tasks_for_target_id[task.target_id] += 1
        if task.target_id not in target_id_to_votes:
            target_id_to_votes[task.target_id] = Counter()
        target_id_to_votes[task.target_id] += Counter(task.selected_questions)

    ### compile result rows and write
    result_rows = []
    for target_id, votes in target_id_to_votes.items():
        valid_votes = [x for x in votes if votes[x] / num_tasks_for_target_id[target_id] >= options.agree_percent]
        result_rows.append([target_id] + valid_votes)
    result_rows = sorted(result_rows, key=lambda point: point[0])

    with open(output_file, 'w+') as f:
        output_csv = csv.writer(f)
        for row in result_rows:
            output_csv.writerow(row)

    ### View the results
    question_file = "../dataset/raw/questions_kway.tsv"
    questions = {}
    ### Load questions
    with open(question_file, 'r') as f:
        question_tsv = csv.reader(f, delimiter = '\t')
        for row in question_tsv:
            questions[int(row[0])] = row[1]

    ### Load original choices
    original_choices = {}
    with open(choices_file, 'r') as f:
        choices_csv = csv.reader(f)
        for row in choices_csv:
            original_choices[int(row[0])] = [int(x) for x in row[1:]]

    rank_total = 0.0
    num_target_questions = 0
    processed_csv_rows = []
    with open(output_file, 'r') as f:
        processed_csv_rows = list(csv.reader(f))

    if options.is_view:
        for row in processed_csv_rows:
            print("#"*80)
            print(questions[int(row[0])])
            print("="*80)
            match_set = set([int(x) for x in row[1:]])
            for i in range(10):
                candidate_id = int(original_choices[int(row[0])][i])
                if candidate_id in match_set:
                    print("-X-\t" + Fore.YELLOW + questions[candidate_id] + Style.RESET_ALL)
                else:
                    print(questions[candidate_id])

    ### Calculate MRR (Mean reciprocal rank)
    for row in processed_csv_rows:
        match_set = set([int(x) for x in row[1:]])
        if len(match_set) > 0:
            rank_total += 1.0 / get_rank(original_choices[int(row[0])], match_set)
        num_target_questions += 1

    print("Mean Reciprocal Rank:\t{}".format(rank_total / num_target_questions))

def process_row(row, turker_index, input_start, result_start):
    turk_id = row[turker_index]
    tasks = []
    for i in range(5):
        target_id = int(row[input_start + (i * 22)]) # get each target id
        choice_ids = [int(x) for x in row[input_start + (i * 22) + 2: input_start + ((i + 1) * 22) - 1: 2]]
        chosen_ids = [int(x) for x in row[result_start+i].split('|')] if len(row[result_start+i]) > 0 else []
        new_task = Task(turk_id, target_id, choice_ids, chosen_ids)
        tasks.append(new_task)
    return tasks

def get_bool_array(choice_questions, selected_questions):
    bool_list = []
    for q in choice_questions:
        bool_list.append(q in selected_questions)
    return np.asarray(bool_list)

def compute_num_agree(x, y):
    return np.sum(np.logical_not(np.logical_xor(x, y)))

def get_rank(choices, selection):
    for i, choice_id in enumerate(choices):
        if choice_id in selection:
            return i + 1
    return 0

if __name__ == "__main__":
    main()
