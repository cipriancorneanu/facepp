__author__ = 'cipriancorneanu'

import tensorflow as tf
from facepp.reader.generator import GeneratorBP4D
import numpy as np
import argparse
import cPickle

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--type")
parser.add_argument("--n_folds", type=int)
parser.add_argument("--test_fold", type=int)
args = parser.parse_args()

# Load models
fname = args.n_folds + 'folds_' + 'tf_' + args.type + '_augm_1'
model_saver = tf.train.import_meta_graph(args.path + 'models/' + fname + '.meta')
model_graph = tf.get_default_graph()

# Show nodes in graph
for op in tf.get_default_graph().get_operations():
    print str(op.name)

# Define input nodes
ims = model_graph.get_tensor_by_name("input/input_images:0")
labels = model_graph.get_tensor_by_name("input/input_images:0")

# Define node to retrieve
feat = model_graph.get_tensor_by_name("input/input_images:0")

# Data
dtg = GeneratorBP4D(args.path, type=[args.type], n_folds=args.n_folds)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
train_folds = list(set(range(1, args.n_folds+1)) - set([args.test_fold]))
test_folds = [args.test_fold]

''' If training on augmented dataset multiply number of samples'''
if args.augm:
    mul_samples = 4
else:
    mul_samples = 1

n_samples_train = mul_samples*np.sum([dtg.n_samples_fold(f) for f in train_folds])
n_samples_test = dtg.n_samples_fold(args.test_fold)


print('Classification on BP4D. Cross-validation.')

l_rate = 0.0005
augmentation = 1
n_classes = 12


# Train session
config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config=config) as sess:
    # Restore model weights from previously saved model
    sess.run(init)

    # Restore model
    saver.restore(sess, args.path+'models/035-5')

    # Test restored model
    iter, feats = (0, [], [], [])
    for f in train_folds+test_folds:
        for _, (images, aus) in enumerate(dtg.generate(fold=f, batch_size=args.batch_size, verbose=False, with_labels=True, augment=False)):
            images, aus = (np.asarray(images, dtype=np.float32), np.asarray(aus, dtype=np.float32))
            images = images/255.

            feat_val   = sess.run(
                [feat],
                feed_dict={
                    ims: images,
                    labels: aus
                    })

            feats.append(feat_val)

            if iter%100==0: print 'Iteration {}'.format(iter)
            '''if iter>100: break'''

            if iter>n_samples_test/args.batch_size: break
            iter = iter+1


    # Save feats
    cPickle.dump(np.concatenate(feats), open(args.path+'features/'+
                                             str(args.n_folds)+'folds_tf_'+str(args.test_fold)+'_'+
                                             args.type+'_augm_'+str(args.augm)+'.pkl', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)
