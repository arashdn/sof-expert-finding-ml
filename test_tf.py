# import tensorflow as tf
#
# # v1 = tf.Variable([[1,2,3]], name="v1")
# # v2 = tf.Variable([[4],[5],[6]], name="v2")
# # ...
# # # Add an op to initialize the variables.
# # init_op = tf.global_variables_initializer()
# #
# # # Add ops to save and restore all the variables.
# # saver = tf.train.Saver()
#
# # with tf.Session() as sess:
# #   sess.run(init_op)
# #   # Do some work with the model.
# #   v2 = tf.matmul(v1,v2)
# #
# #   # print(v2.eval())
# #
# #   # Save the variables to disk.
# #   save_path = saver.save(sess, "./model.ckpt")
# #   print("Model saved in file: %s" % save_path)
# #
#
# # Create some variables.
# v1 = tf.Variable([[1, 2, 3]], name="v1")
# v2 = tf.Variable([0], name="v2")
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, "./model.ckpt")
#     print("Model restored.")
#
#     print(v2.eval())



import numpy as np
a = np.load("./save/wp.npy")
np.savetxt("./save/wp22.txt",a)
