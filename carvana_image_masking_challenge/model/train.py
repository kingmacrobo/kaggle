import tensorflow as tf
import unet
import data

flags = tf.app.flags

flags.DEFINE_string('model_dir', 'checkpoints', 'the checkpoints directory')
flags.DEFINE_string('out_mask_dir', 'out_mask', 'the mask result output directory')
flags.DEFINE_string('train_list', '', 'the file contains train list')
flags.DEFINE_string('test_list', '', 'the file contains test list')
flags.DEFINE_string('train_mask_dir', '', 'the train mask directory')
flags.DEFINE_string('debug_dir', '', 'the debug directory')

FLAGS = flags.FLAGS

def main():
    #datagen = data.DataGenerator(FLAGS.train_list, FLAGS.test_list, FLAGS.train_mask_dir, debug_dir=FLAGS.debug_dir)
    datagen = data.DataGenerator(FLAGS.train_list, FLAGS.test_list, FLAGS.train_mask_dir)
    model = unet.UNET(datagen, out_mask_dir=FLAGS.out_mask_dir, model_dir=FLAGS.model_dir)
    with tf.Session() as session:
        model.train(session)

if __name__ == '__main__':
    main()
