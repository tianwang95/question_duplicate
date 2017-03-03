import tensorflow as tf
import numpy as np
import os
import sys
import shutil
from keras import backend as K
from models import concat_gru
from data import Data
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib
from keras.models import load_model, model_from_config, model_from_json

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    K.set_learning_phase(0)

    ### Clear out directories
    export_dir = "export/concat_gru_model_2hid"
    temp_dir = "tmp/"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    ### Set variables for freeze step
    checkpoint_name = "saved_checkpoint"
    checkpoint_prefix = os.path.join(temp_dir, checkpoint_name)
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "model.pb"
    input_graph_path = os.path.join(temp_dir, input_graph_name)
    input_checkpoint_path = os.path.join(temp_dir, checkpoint_name)
    output_graph_path = os.path.join(export_dir, output_graph_name)

    ### Load data and model
    data = Data("dataset/raw/quora_duplicate_questions.tsv",
                embed_dim=100,
                batch_size=64,
                dev_mode=False)

    model = concat_gru.get_model(
                data,
                dim = 128,
                weights = data.embedding_matrix,
                dropout_W = 0.2,
                dropout_U = 0.2,
                num_hidden=2)

    model.load_weights("saved_models/concat_gru_2hid/model.09-0.83.hdf5")

    ### eval
#    print('Evaluating')
#    print(model.evaluate_generator(data.dev_generator(), data.dev_count))
#    print(model.metrics_names)
#    return

    #### Save the model to a checkpoint
    sess = K.get_session()
    saver = saver_lib.Saver()
    checkpoint_path = saver.save(
            sess,
            checkpoint_prefix,
            global_step=0,
            latest_filename=checkpoint_state_name)

    ### Save graph definition
    tf.train.write_graph(sess.graph.as_graph_def(), temp_dir, input_graph_name)
    
    ### Freeze graph
    output_node_names = "Sigmoid"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    input_saver_def_path = ""
    input_binary = False
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path,
                              input_saver_def_path,
                              input_binary,
                              checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_path,
                              clear_devices,
                              "")

    vocab_list_name = "model_params.txt"
    q1_input_name = "q1_input"
    q2_input_name = "q2_input"
    with open(os.path.join(export_dir, vocab_list_name), 'w') as f:
        f.write(str(data.max_sentence_length) + '\n')
        f.write(q1_input_name + ':0\n')
        f.write(q2_input_name + ':0\n')
        f.write(output_node_names + ':0\n')
        for word in data.vocabulary:
            f.write(word + '\n')

    # remove the temp directory
    shutil.rmtree(temp_dir)

    #### Test to make sure it has worked
    graph = load_graph(output_graph_path)
    q1_input = graph.get_tensor_by_name('import/q1_input:0')
    q2_input = graph.get_tensor_by_name('import/q2_input:0')
    y_tensor = graph.get_tensor_by_name('import/Sigmoid:0')
    print("Evaluating...")

    with tf.Session(graph = graph) as session:
        acc = 0.0;
        count = 0
        for point in data.dev_generator():
            if count >= 400:
                break
            x1 = point[0][0]
            x2 = point[0][1]
            y = point[1]
            y_out = session.run(y_tensor, feed_dict = {
                    q1_input: x1,
                    q2_input: x2,
                })
            count += 1
            acc += np.sum(y == np.round(y_out)) / 64.0
            print(acc / float(count))
        print("Final acc: {}".format(acc / float(count)))

def question_to_indices(question, data):
    q_id = 999999
    data.add_question(q_id, question)
    data.transform_to_embed(q_id, data.questions[q_id])
    return data.questions_embed[q_id]

def load_graph(output_graph_path):
    with tf.gfile.GFile(output_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

main()
