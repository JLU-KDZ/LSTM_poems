import os
import collections
import numpy as np
import tensorflow as tf
import argparse

from lstm import rnn_model
from process import data_process, generate_batch

batch_size = 1

# 向量转换为对应字符
def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

# 生成诗词 type必须为5／7
def get_pomes(heads, type):
    if type != 5 and type != 7:
        print('第二个参数为5或者7，请填写对应值')
        return
    poems_vector, word_int_map, vocabularies = data_process('./data/poetry.txt')
    input_data = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    learning_rate = tf.Variable(0.0, trainable=False)

    end_points = rnn_model(model='lstm', input_data=input_data, batch_size=batch_size, output_data=None,
                           vocab_size=len(vocabularies))

    Session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=Session_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, './model/poems-0')

        poem = ''
        for head in heads:
            flag = True
            while flag:
                x = np.array([list(map(word_int_map.get, '['))])
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x})

                sentence = head
                x = np.zeros((1, 1))
                x[0, 0] = word_int_map[sentence]
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                word = to_word(predict, vocabularies)
                sentence += word

                while word != '。':
                    x = np.zeros((1, 1))
                    x[0, 0] = word_int_map[word]
                    [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                     feed_dict={input_data: x, end_points['initial_state']: last_state})
                    word = to_word(predict, vocabularies)
                    sentence += word
                if len(sentence) == 2 + 2 * type:
                    sentence += '\n'
                    poem += sentence
                    flag = False
        return poem

def main(_):
    parser = argparse.ArgumentParser(description="Generate a Chinese acrostic poem.")
    parser.add_argument("type", type=int, choices=[5, 7], help="Specify the poem type: 5 for five characters per line, 7 for seven characters per line.")
    parser.add_argument("heads", type=str, help="The head content for the acrostic poem.")
    args = parser.parse_args()

    print(get_pomes(args.heads, args.type))


if __name__ == '__main__':
    tf.compat.v1.app.run()
