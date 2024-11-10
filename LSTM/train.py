

import os
import numpy as np
import tensorflow as tf

from lstm import rnn_model
from process import data_process, generate_batch

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.02, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./model'), 'model save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poetry.txt'), 'file name of poems.')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 15, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS
    
def run_training():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
        
    # 将文件转换为词向量
    poems_vector, word_to_int, vocabularies = data_process(FLAGS.file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
    
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    learning_rate = tf.Variable(0.0, trainable=False)  
    
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=learning_rate)
    
    Session_config = tf.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True 
    
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=Session_config) as sess:
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                # 学习率随训练次数不断减少
                sess.run(tf.assign(learning_rate, FLAGS.learning_rate * (0.97 ** epoch))) 
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    if (batch % 50 ==0):
                        print('epoch: %d, batch: %d, loss: %.6f, learning_rate: %.6f' % (epoch, batch, loss, FLAGS.learning_rate * (0.97 ** epoch)))
                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))
    
def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()