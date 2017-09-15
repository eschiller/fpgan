import sys
sys.path.append("./data")
from fpdatamgr import fpdatamgr
import fpdata
import tensorflow as tf
import numpy as np
import tf_utils




class fp_gan_nn:
    def __init__(self,
                 batch_size=100,
                 learn_rate_dn=0.0001,
                 learn_rate_gn=0.0001,
                 np_x_dim=64,
                 np_y_dim=64,
                 train_data_size=20000,
                 optimizer=tf.train.AdamOptimizer,
                 sample_label="test",
                 debug=False):

        self.np_x_dim = np_x_dim
        self.np_y_dim = np_y_dim
        self.debug = debug
        self.sample_label = sample_label
        #make sure printing numpy arrays is in full
        np.set_printoptions(threshold=np.nan)

        #initialize some vars
        self.batch_size = batch_size
        self.train_data_size=train_data_size
        self.datamgr = fpdatamgr(np_x_dim=self.np_x_dim, np_y_dim=self.np_y_dim)
        self.datamgr.import_json_file("./data/json/datafp.json")
        self.fp_data = self.datamgr.generate_data_set(self.train_data_size)

        #uncomment to use a simple single floorplan for entire dataset
        #self.fp_data = self.datamgr.generate_test_set(self.train_data_size)

        #VARIABLES
        self.w_gn_h1 = tf_utils.weight_var([100, 1024], name="gen_w1")
        self.w_gn_h2 = tf_utils.weight_var([1024, 128 * self.np_x_dim * self.np_y_dim], name="gen_w2")
        self.w_gn_h3 = tf_utils.weight_var([5, 5, 64, 128], name="gen_w3")
        self.w_gn_h4 = tf_utils.weight_var([5, 5, 2, 64])

        self.w_dn_h1 = tf_utils.weight_var([1, 1, 2, 8], name="discrim_w1")
        self.w_dn_h2 = tf_utils.weight_var([5, 5, 8, 16], name="discrim_w1")
        self.w_dn_h3 = tf_utils.weight_var([16 * self.np_x_dim * self.np_y_dim, 1024])

        #INPUT PARAMS
        self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, 100])
        self.real_x = tf.placeholder(tf.float32, shape=[self.batch_size, self.np_x_dim, self.np_y_dim, 2])

        #####
        #COST AND TRAINING
        #####

        #GN COST/TRAINING

        self.gen_y = self.gn(self.noise)
        self.gen_y_sig = tf.nn.sigmoid(self.gen_y)
        self.raw_real = self.dn(self.real_x)
        self.p_real = tf.nn.sigmoid(self.raw_real)

        #todo you could try not running gen_y through sigmoid
        self.raw_gen = self.dn(self.gen_y_sig)
        self.p_gen = tf.nn.sigmoid(self.raw_gen)


        #DN COST/TRAINING
        self.ce_dn_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_real, labels=tf.ones_like(self.raw_real)))
        self.ce_dn_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_gen, labels=tf.zeros_like(self.raw_gen)))
        self.ce_dn = self.ce_dn_real + self.ce_dn_gen
        self.train_step_dn = optimizer(learn_rate_dn, beta1=0.5).minimize(self.ce_dn,var_list=[self.w_dn_h1, self.w_dn_h2, self.w_dn_h3])

        self.ce_gn = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_gen, labels=tf.ones_like(self.raw_gen)))

        #todo uncomment below
        self.train_step_gn = optimizer(learn_rate_gn, beta1=0.5).minimize(self.ce_gn, var_list=[self.w_gn_h1, self.w_gn_h2, self.w_gn_h3, self.w_gn_h4])

        #PREP
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        print(self.sess.run(init))



    def gn(self, Z):
        '''
        Accept [batch_size, 100] of noise and outputs [batch_size, self.np_x_dim, self.np_y_dim, 2] of generated floorplans
        :param Z:
        :return:
        '''
        # matmul [100,100] * [100,1024], resulting in [100,1024]
        h1 = tf.nn.relu(tf.matmul(Z, self.w_gn_h1))

        #gen_W2 is [100,1024] * [1024,524288] resulting in [100,524288 (128 * 2 * 2)]
        h2 = tf.nn.relu(tf.matmul(h1, self.w_gn_h2))

        # this will result in [100,self.np_x_dim,self.np_y_dim,128]
        h2 = tf.reshape(h2, [self.batch_size,self.np_x_dim,self.np_y_dim,128])

        #output shape is [100,self.np_x_dim,self.np_y_dim,64]
        output_shape_l3 = [self.batch_size,self.np_x_dim,self.np_y_dim,64]

        #[100,self.np_x_dim,self.np_y_dim,128] with filters of [5,5,64,128] at [1,1,1,1] strides
        #will have output shape of [100,self.np_x_dim,self.np_y_dim,64] after transpose
        h3 = tf.nn.conv2d_transpose(h2, self.w_gn_h3, output_shape=output_shape_l3, strides=[1,1,1,1])
        h3 = tf.nn.relu(h3)

        # output shape is [100,self.np_x_dim,self.np_y_dim,2]
        output_shape_l4 = [self.batch_size,self.np_x_dim,self.np_y_dim,2]

        #[100,self.np_x_dim,self.np_y_dim,64] with filters at [5,5,2,64] and strides [1,1,1,1]
        #will have [100,self.np_x_dim,self.np_y_dim,2] after transpose
        h4 = tf.nn.conv2d_transpose(h3, self.w_gn_h4, output_shape=output_shape_l4, strides=[1,1,1,1])

        return h4



    def dn(self, fpdata):
        '''
        Accept [batch_size, self.np_x_dim, self.np_y_dim, 2] and outputs [batch_size, 1024]. The 1024 is the classification
        :param fpdata:
        :return:
        '''
        # [100, self.np_x_dim, self.np_y_dim, 2] with filters [1,1,2,8] will output [100, self.np_x_dim, self.np_y_dim, 8]
        h1 = tf.nn.relu( tf.nn.conv2d( fpdata, self.w_dn_h1, strides=[1,1,1,1], padding='SAME' ))

        #conv2d([100, self.np_x_dim, self.np_y_dim, 8] with filters [5, 5, 8, 16]) = [100, self.np_x_dim, self.np_y_dim, 16]
        h2 = tf.nn.relu( tf.nn.conv2d( h1, self.w_dn_h2, strides=[1,1,1,1], padding='SAME') )

        #reshape for [100, self.np_x_dim * self.np_y_dim * 16]
        h2 = tf.reshape(h2, [self.batch_size, -1])

        #[100, self.np_x_dim * self.np_y_dim * 16] * [self.np_x_dim * self.np_y_dim * 16, 1024] = [100, 1024]
        h3 = tf.nn.relu(tf.matmul(h2, self.w_dn_h3))

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

        #If we're going to go past the size of the training data, we'll loop back to the beginning with a modulus op
        scratch, start = divmod((rep * size) + size, self.train_data_size)
        end = start + size
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

                self.train_gn(self.batch_size, check_acc=True)
                self.train_dn(i, self.batch_size, check_acc=True)

            self.train_dn(i, size=self.batch_size)
            self.train_gn(size=self.batch_size)



    def generate(self, size=100):
        feed_noise = np.random.uniform(-1, 1, size=[size, 100]).astype(np.float32)
        return self.sess.run(self.gen_y_sig, feed_dict={self.noise: feed_noise})


    def save_checkpoint(self, reps):
        fp_samples = self.generate(self.batch_size)
        rescaled_samples = fpdata.np_rescale(fp_samples, snap=False)
        sample_to_out = rescaled_samples[0]

        #uncomment below to get printouts of sample
        if self.debug == True:
            print(sample_to_out)

        self.datamgr.import_sample_fp(sample_to_out)
        self.datamgr.export_svg(-1, "./samples/" + self.sample_label + "_" + str(reps) + ".svg")

