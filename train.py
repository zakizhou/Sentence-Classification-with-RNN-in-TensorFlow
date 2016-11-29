from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from model import TextModel, Config
from input import Inputs
import tensorflow as tf


def main():
    with tf.Graph().as_default():
        config = Config()
        with tf.name_scope("train"):
            inputs = Inputs(batch_size=84, capacity=4000, train=True)
            with tf.variable_scope("model", reuse=None):
                m = TextModel(config, inputs)

        with tf.name_scope("valid"):
            valid_inputs = Inputs(batch_size=84, capacity=800, train=False)
            with tf.variable_scope("model", reuse=True):
                mvalid = TextModel(config, valid_inputs)

        sess = tf.Session()
        init = tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            index = 0
            while not coord.should_stop():
                _, loss_value = sess.run([m.train_op, m.loss])
                print("step %d, loss %f" % (index, loss_value))
                if (index + 1) % 5 == 0:
                    valid_accuracy = sess.run(mvalid.validation)
                    print("validation accuracy %f" % valid_accuracy)
                index += 1
        except tf.errors.OutOfRangeError:
            print("epoch limits reached, stopping threads!")
        except KeyboardInterrupt:
            print("keyboard interrupt detected, stopping threads!")
            del sess
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
        del sess


if __name__ == "__main__":
    main()
