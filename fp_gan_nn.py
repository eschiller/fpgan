import sys
sys.path.append("./data")
from fpdatamgr import fpdatamgr
import tensorflow as tf
import numpy as np
import tf_utils




class fp_gan_nn:
    def __init__(self, batch_size = 100, learn_rate_dn=0.0001, learn_rate_gn=0.0001, optimizer=tf.train.AdamOptimizer):
        np.set_printoptions(threshold=np.nan)
        self.batch_size = batch_size
        #initialize some vars
        self.fp_data = fpdatamgr().generate_test_set(20000)

        #INPUTS/PARAMS
        #self.real_x_sig = tf.nn.sigmoid(self.real_x)



        #VARIABLES
        self.w_gn_h1 = tf_utils.weight_var([100, 1024], name="gen_w1")
        self.w_gn_h2 = tf_utils.weight_var([1024, 128*2*2], name="gen_w2")
        self.w_gn_h3 = tf_utils.weight_var([5, 5, 64, 128], name="gen_w3")
        self.w_gn_h4 = tf_utils.weight_var([5, 5, 5, 64])

        self.w_dn_h1 = tf_utils.weight_var([5, 5, 5, 8], name="discrim_w1")
        self.w_dn_h2 = tf_utils.weight_var([5, 5, 8, 16], name="discrim_w1")
        self.w_dn_h3 = tf_utils.weight_var([16*8*8, 1024])

        #INPUT PARAMS

        self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, 100])
        self.real_x = tf.placeholder(tf.float32, shape=[self.batch_size, 8, 8, 5])

        #####
        #COST AND TRAINING
        #####

        #GN COST/TRAINING

        self.gen_y = self.gn(self.noise)
        self.gen_y_sig = tf.nn.sigmoid(self.gen_y)
        self.raw_real = self.dn(self.real_x)
        self.p_real = tf.nn.sigmoid(self.raw_real)
        self.raw_gen = self.dn(self.gen_y_sig)
        self.p_gen = tf.nn.sigmoid(self.raw_gen)


        #DN COST/TRAINING
        self.ce_dn_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_real, labels=tf.ones_like(self.raw_real)))
        self.ce_dn_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_gen, labels=tf.zeros_like(self.raw_gen)))
        self.ce_dn = self.ce_dn_real + self.ce_dn_gen
        self.train_step_dn = optimizer(learn_rate_dn, beta1=0.5).minimize(self.ce_dn,var_list=[self.w_dn_h1, self.w_dn_h2, self.w_dn_h3])

        self.ce_gn = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_gen, labels=tf.ones_like(self.raw_gen)))

        self.train_step_gn = optimizer(learn_rate_gn, beta1=0.5).minimize(self.ce_gn, var_list=[self.w_gn_h1, self.w_gn_h2, self.w_gn_h3, self.w_gn_h4])

        #PREP
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)



    def gn(self, Z):
        #todo change size of Z to more closely match other (100)
        # matmul [100, 100] * [100, 1024], resulting in [100, 1024]
        h1 = tf.nn.relu(tf.matmul(Z, self.w_gn_h1))

        #gen_W2 is [100, 1024] * [1024, 512] resulting in [100, 512 (128 * 2 * 2)]
        h2 = tf.nn.relu(tf.matmul(h1, self.w_gn_h2))

        # this will result in [100, 2, 2, 128]
        h2 = tf.reshape(h2, [self.batch_size,2,2,128])

        #output shape is [100, 4, 4, 64]
        output_shape_l3 = [self.batch_size,4,4,64]

        #well, here I guess we've got [100, 7, 7, 138] * [5, 5, 64, 138] = [100, 14, 14, 64]? Pretty sure we're just making shit up now.
        h3 = tf.nn.conv2d_transpose(h2, self.w_gn_h3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu(h3)

        # output shape is [100, 8, 8, 5]
        output_shape_l4 = [self.batch_size,8,8,5]

        #ok, so [100, 14, 14, 74] * [5, 5, 1, 74] resulting in [100, 28, 28, 1] after transpose
        h4 = tf.nn.conv2d_transpose(h3, self.w_gn_h4, output_shape=output_shape_l4, strides=[1,2,2,1])

        # I don't think we need to reshape here. We'll just output in the correct shape
        #and return the [100, 28, 28, 1]
        #h4 = tf.reshape(h4, [100, 784])
        return h4


    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, 100])

        h1 = tf.nn.relu(tf.matmul(Z, self.w_gn_h1))

        #gen_W2 is [100, 1024] * [1024, 6272] resulting in [100, 6272 (128 * 7 * 7)]
        h2 = tf.nn.relu(tf.matmul(h1, self.w_gn_h2))

        # this will result in [100, 7, 7, 128]
        h2 = tf.reshape(h2, [self.batch_size,7,7,128])

        #output shape is [100, 14, 14, 64]
        output_shape_l3 = [self.batch_size,14,14,64]

        #well, here I guess we've got [100, 7, 7, 138] * [5, 5, 64, 138] = [100, 14, 14, 64]? Pretty sure we're just making shit up now.
        h3 = tf.nn.conv2d_transpose(h2, self.w_gn_h3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu(h3)

        # output shape is [100, 28, 28, 1]
        output_shape_l4 = [self.batch_size,28,28,1]

        #ok, so [100, 14, 14, 74] * [5, 5, 1, 74] resulting in [100, 28, 28, 1] after transpose
        h4 = tf.nn.conv2d_transpose(h3, self.w_gn_h4, output_shape=output_shape_l4, strides=[1,2,2,1])

        #and return the [100, 28, 28, 1]
        h4 = tf.reshape(h4, [100, 784])
        x = tf.nn.sigmoid(h4)
        return Z,x



    def dn(self, fpdata):

        # Can do a reshape here if needed, but since we're planning on working directly in 8x8x5 I don't think
        # We need to
        #[100, 28, 28, 1]
        #X = tf.reshape(image, [100, 28, 28, 1])

        # [100, 8, 8, 5] with filters [5, 5, 5, 8] gonna output [100, 8, 8, 8]
        h1 = tf.nn.relu( tf.nn.conv2d( fpdata, self.w_dn_h1, strides=[1,1,1,1], padding='SAME' ))

        #conv2d([100, 8, 8, 8] with filters [5, 5, 8, 16]) = [100, 8, 8, 16]
        h2 = tf.nn.relu( tf.nn.conv2d( h1, self.w_dn_h2, strides=[1,1,1,1], padding='SAME') )

        #[100, 8*8*16]
        h2 = tf.reshape(h2, [self.batch_size, -1])

        #[100, 8*8*16 + 10] * [8*8*16, 1024] = [100, 1024]
        h3 = tf.nn.relu( tf.matmul(h2, self.w_dn_h3 ) )

        #returns [100, 1024]
        return h3



    def train_gn(self, size=100, check_acc=False):
        feed_noise = np.random.uniform(-1, 1, size=[size, 100]).astype(np.float32)

        if check_acc:
            acc = self.sess.run(self.ce_gn, feed_dict={self.noise: feed_noise})
            print("gn accuracy: " + str(acc))
        else:
            self.sess.run(self.train_step_gn, feed_dict={self.noise: feed_noise})


    def train_dn(self, rep, size=100, check_acc=False):
        feed_noise = np.random.uniform(-1, 1, size=[size, 100]).astype(np.float32)
        #start = (rep % 50) * 100
        start = 0
        end = start + 100
        if check_acc:
            acc, gen_y_sig, gen_y = self.sess.run([self.ce_dn, self.gen_y_sig, self.gen_y], feed_dict={self.real_x: self.fp_data[start:end], self.noise: feed_noise})
            print("dn accuracy: " + str(acc))
            print("gen x avg: " + str(gen_y_sig.mean()))
        else:
            self.sess.run(self.train_step_dn, feed_dict={self.real_x: self.fp_data[start:end], self.noise: feed_noise})


    def train_all(self, reps=1000):
        for i in range(reps):
            _, dont_save = divmod(i, 10)
            if not dont_save:
                self.save_checkpoint(i)
                print("done with round: " + str(i))

                self.train_gn(check_acc=True)
                self.train_dn(i, check_acc=True)

            self.train_dn(i)
            self.train_gn()
            #print("done with round: " + str(i))



    def generate(self, size=100):
        feed_noise = np.random.uniform(-1, 1, size=[size, 100]).astype(np.float32)
        return self.sess.run(self.gen_y_sig, feed_dict={self.noise: feed_noise})


    def save_checkpoint(self, reps):
        fp_samples = self.generate(100)
        print np.rint(fp_samples[0])
        #mnist_imaging.save_image(images, "4gen_cp_reps_" + str(reps) + "_" + "1.jpg")
        #mnist_imaging.save_image(images[1:4], "4gen_cp_reps_" + str(reps) + "_" + "2.jpg")
        #mnist_imaging.save_image(images[2:], "gen_cp_reps_" + str(reps) + "_" + "3")
        #mnist_imaging.save_image(images[3:], "gen_cp_reps_" + str(reps) + "_" + "4")
        #mnist_imaging.save_image(images[4:], "gen_cp_reps_" + str(reps) + "_" + "5")


