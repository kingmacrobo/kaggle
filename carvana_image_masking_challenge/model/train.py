import tensorflow as tf
import fcn
import data

flags = tf.app.flags
flags.DEFINE_string('model_dir', 'checkpoints', 'the checkpoints directory')
flags.DEFINE_string('out_mask_dir', 'out_mask', 'the mask result output directory')
flags.DEFINE_string('train_list_file', '', 'the file contains train list')
flags.DEFINE_string('test_list_file', '', 'the file contains test list')
flags.DEFINE_string('train_mask_dir', '', 'the train mask directory')

def main():
    datagen = data.DataGenerator(flags.train_list_file, flags.test_list_file, flags.train_mask_dir)
    fcn_model = fcn.FCN(datagen, out_mask_dir='validate_out_mask')
    with tf.Session as session:
        fcn_model.train(session)

if __name__ == '__main__':
    tf.app.run()
