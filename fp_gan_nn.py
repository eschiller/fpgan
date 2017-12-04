import sys
sys.path.append("./data")
sys.path.append("./metrics")
from fpdatamgr import fpdatamgr
from metricmgr import metricmgr
import fpdata
import tensorflow as tf
import numpy as np
import tf_utils




class fp_gan_nn:
    def __init__(self,
                 batch_size=100,
                 learn_rate_dn=0.0002,
                 learn_rate_gn=0.0002,
                 np_x_dim=64,
                 np_y_dim=64,
                 train_data_size=20000,
                 optimizer=tf.train.AdamOptimizer,
                 sample_label="test",
                 sample_data=False,
                 debug=False,
                 restore_checkpoint=None,
                 ):

        self.np_x_dim = np_x_dim
        self.np_y_dim = np_y_dim
        self.debug = debug
        self.sample_label = sample_label
        self.sample_data = sample_data
        #make sure printing numpy arrays is in full
        np.set_printoptions(threshold=np.nan)
        self.metmgr = metricmgr()

        #initialize some vars
        self.dropout = .75
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = batch_size
        self.train_data_size=train_data_size
        self.datamgr = fpdatamgr(np_x_dim=self.np_x_dim, np_y_dim=self.np_y_dim)


        #uncomment to use the "real" dataset from the json file
        self.datamgr.import_json_file("./data/json/datafp.json")
        #self.fp_data = self.datamgr.generate_data_set(self.train_data_size, generations=10, rnd_rescale=True)
        self.fp_data = self.datamgr.generate_data_set(self.train_data_size, generations=10, rnd_reflect=True)
        #self.fp_data = self.datamgr.generate_data_set(self.train_data_size)

        #uncomment to use test floorplan (square with line through it) for entire dataset
        #self.fp_data = self.datamgr.generate_test_set(self.train_data_size)
        
        #uncomment to use a simple single floorplan for entire dataset
        #self.fp_data = self.datamgr.generate_svg_test_set("./data/vec/3.svg", 100000)


        #VARIABLES
        self.w_gn_h0 = tf_utils.weight_var([100, 256 * self.np_x_dim * self.np_y_dim], name="gen_w0")
        self.w_gn_h1 = tf_utils.weight_var([5, 5, 128, 256], name="gen_w1")
        self.w_gn_h2 = tf_utils.weight_var([9, 9, 64, 128], name="gen_w2")
        self.w_gn_h3 = tf_utils.weight_var([1, 1, 2, 64], name="gen_w3")
        self.gn_h0_bias = tf.Variable(tf.zeros([256]), name="dn_h0_bias")
        self.gn_h1_bias = tf.Variable(tf.zeros([128]), name="dn_h1_bias")
        self.gn_h2_bias = tf.Variable(tf.zeros([64]), name="dn_h2_bias")
        self.gn_h3_bias = tf.Variable(tf.zeros([2]), name="dn_h3_bias")

        self.w_dn_h1 = tf_utils.weight_var([1, 1, 2, 16], name="discrim_w1")
        self.w_dn_h2 = tf_utils.weight_var([9, 9, 16, 32], name="discrim_w2")
        self.w_dn_h3 = tf_utils.weight_var([5, 5, 32, 64], name="discrim_w3")
        self.w_dn_h4 = tf_utils.weight_var([64 * self.np_x_dim * self.np_y_dim, 512], name="discrim_w4")
        self.dn_h1_bias = tf.Variable(tf.zeros([16]), name="dn_h1_bias")
        self.dn_h2_bias = tf.Variable(tf.zeros([32]), name="dn_h2_bias")
        self.dn_h3_bias = tf.Variable(tf.zeros([64]), name="dn_h3_bias")
        self.dn_h4_bias = tf.Variable(tf.zeros([512]), name="dn_h4_bias")

        #INPUT PARAMS
        self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, 100])
        self.real_x = tf.placeholder(tf.float32, shape=[self.batch_size, self.np_x_dim, self.np_y_dim, 2])

        #####
        #COST AND TRAINING
        #####

        #GN COST/TRAINING
        self.gen_y = self.gn(self.noise)
        self.gen_y_sig = tf.nn.sigmoid(self.gen_y)
        self.raw_gen_logits, self.raw_gen_act = self.dn(self.gen_y_sig)


        #DN COST/TRAINING
        self.raw_real_logits, self.raw_real_act = self.dn(self.real_x)
        self.ce_dn_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_real_logits, labels=tf.ones_like(self.raw_real_logits)))
        self.ce_dn_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_gen_logits, labels=tf.zeros_like(self.raw_gen_logits)))
        self.ce_dn = self.ce_dn_real + self.ce_dn_gen
        self.train_step_dn = optimizer(learn_rate_dn, beta1=0.5).minimize(self.ce_dn,var_list=[self.dn_h1_bias, self.dn_h2_bias, self.dn_h3_bias, self.dn_h4_bias, self.w_dn_h1, self.w_dn_h2, self.w_dn_h3])

        #GN COST/TRAINING
        self.ce_gn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_gen_logits, labels=tf.ones_like(self.raw_gen_logits)))
        self.train_step_gn = optimizer(learn_rate_gn, beta1=0.5).minimize(self.ce_gn, var_list=[self.gn_h0_bias, self.gn_h1_bias, self.gn_h2_bias, self.gn_h3_bias, self.w_gn_h1, self.w_gn_h2, self.w_gn_h3])

        #PREP
        #uncomment to see hardware device (gpu) placement
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        print(self.sess.run(init))

        if restore_checkpoint:
            self.saver.restore(self.sess, restore_checkpoint)

    def gn(self, Z):
        '''
        Accept [batch_size, 100] of noise and outputs [batch_size, self.np_x_dim, self.np_y_dim, 2] of generated floorplans
        :param Z:
        :return:
        '''
        # matmul [100,100] * [100,1024], resulting in [100,1024]
        h0 = tf.nn.relu(tf.matmul(Z, self.w_gn_h0))
        # this will result in [100,self.np_x_dim,self.np_y_dim,128]
        h0 = tf.reshape(h0, [self.batch_size,self.np_x_dim,self.np_y_dim,256])
        h0 = tf.nn.bias_add(h0, self.gn_h0_bias)

        output_shape_l1 = [self.batch_size,self.np_x_dim,self.np_y_dim,128]
        h1 = tf.nn.conv2d_transpose(h0, self.w_gn_h1, output_shape=output_shape_l1,strides=[1,1,1,1])
        h1 = tf.nn.bias_add(h1, self.gn_h1_bias)
        h1 = tf.nn.relu(h1)

        #output shape is [100,self.np_x_dim,self.np_y_dim,64]
        output_shape_l2 = [self.batch_size,self.np_x_dim,self.np_y_dim,64]

        #[100,self.np_x_dim,self.np_y_dim,128] with filters of [5,5,64,128] at [1,1,1,1] strides
        #will have output shape of [100,self.np_x_dim,self.np_y_dim,64] after transpose
        h2 = tf.nn.conv2d_transpose(h1, self.w_gn_h2, output_shape=output_shape_l2, strides=[1,1,1,1])
        h2 = tf.nn.bias_add(h2, self.gn_h2_bias)
        h2 = tf.nn.relu(h2)

        # output shape is [100,self.np_x_dim,self.np_y_dim,2]
        output_shape_l3 = [self.batch_size,self.np_x_dim,self.np_y_dim,2]

        #[100,self.np_x_dim,self.np_y_dim,64] with filters at [5,5,2,64] and strides [1,1,1,1]
        #will have [100,self.np_x_dim,self.np_y_dim,2] after transpose
        h3 = tf.nn.conv2d_transpose(h2, self.w_gn_h3, output_shape=output_shape_l3, strides=[1,1,1,1])
        h3 = tf.nn.bias_add(h3, self.gn_h3_bias)

        return h3



    def dn(self, fpdata):
        '''
        Accept [batch_size, self.np_x_dim, self.np_y_dim, 2] and outputs [batch_size, 1024]. The 1024 is the classification
        :param fpdata:
        :return:
        '''
        # [100, self.np_x_dim, self.np_y_dim, 2] with filters [1,1,2,8] will output [100, self.np_x_dim, self.np_y_dim, 8]
        h1 = tf.nn.relu( tf.nn.conv2d( fpdata, self.w_dn_h1, strides=[1,1,1,1], padding='SAME' ))
        h1 = tf.nn.bias_add(h1, self.dn_h1_bias)
        h1 = tf.nn.dropout(h1, self.keep_prob)

        #conv2d([100, self.np_x_dim, self.np_y_dim, 8] with filters [5, 5, 8, 16]) = [100, self.np_x_dim, self.np_y_dim, 16]
        h2 = tf.nn.relu( tf.nn.conv2d( h1, self.w_dn_h2, strides=[1,1,1,1], padding='SAME') )
        h2 = tf.nn.bias_add(h2, self.dn_h2_bias)
        h2 = tf.nn.dropout(h2, self.keep_prob)

        #reshape for [100, self.np_x_dim * self.np_y_dim * 16]
        h3 = tf.nn.relu( tf.nn.conv2d( h2, self.w_dn_h3, strides=[1,1,1,1], padding='SAME') )
        h3 = tf.nn.bias_add(h3, self.dn_h3_bias)
        h3 = tf.nn.dropout(h3, self.keep_prob)

        h3 = tf.reshape(h3, [self.batch_size, -1])

        #[100, self.np_x_dim * self.np_y_dim * 16] * [self.np_x_dim * self.np_y_dim * 16, 1024] = [100, 1024]
        #h3 = tf.nn.relu(tf.matmul(h2, self.w_dn_h3))
        h4_logits = tf.matmul(h3, self.w_dn_h4)
        h4_logits = tf.nn.bias_add(h4_logits, self.dn_h4_bias)

        h4_act = tf.nn.sigmoid(h4_logits)
        #returns [100, 1024]
        return h4_logits, h4_act



    def train_gn(self, size=100, check_acc=False):
        '''
        Trains the GN with a batch of the passed size, first generating noise
        to feed the GN, generating floorplans, then feeding those to the dn and
        checking for accuracy.

        :param size:
        :param check_acc:
        :return:
        '''
        feed_noise = np.random.uniform(-1, 1, size=[size, 100]).astype(np.float32)

        if check_acc:
            acc = self.sess.run(self.ce_gn, feed_dict={self.noise: feed_noise, self.keep_prob: 1.0})
            print("gn accuracy: " + str(acc))
        else:
            self.sess.run(self.train_step_gn, feed_dict={self.noise: feed_noise, self.keep_prob: 1.0})


    def train_dn(self, rep, size=100, check_acc=False):
        '''
        Trains the DN with a combination of generator network data and dataset
        data.

        :param rep:
        :param size:
        :param check_acc:
        :return:
        '''
        feed_noise = np.random.uniform(-1, 1, size=[size, 100]).astype(np.float32)

        #If we're going to go past the size of the training data, we'll loop back to the beginning with a modulus op
        scratch, start = divmod((rep * size) + size, self.train_data_size)
        end = start + size

        #below three lines output what's happening in the dataset
        feed_data = self.fp_data[start:end]

        if self.sample_data and check_acc:
            rescaled_feed_data = fpdata.np_rescale(feed_data, snap=False)
            self.datamgr.import_sample_fp(rescaled_feed_data[rep % size])
            self.datamgr.export_svg(-1, "./samples/" + "dataset_" + str(rep) + ".svg")

        if check_acc:
            acc, gen_y_sig, gen_y = self.sess.run([self.ce_dn, self.gen_y_sig, self.gen_y], feed_dict={self.real_x: feed_data, self.noise: feed_noise, self.keep_prob: 1.0})
            print("dn accuracy: " + str(acc))
            print("gen x avg: " + str(gen_y_sig.mean()))
        else:
            self.sess.run(self.train_step_dn, feed_dict={self.real_x: feed_data, self.noise: feed_noise, self.keep_prob: self.dropout})


    def train_all(self, reps=1000):
        '''
        wrapper around train_dn and train_gn. Will force accuracy check and
        save a sample every 10 reps.
        :param reps:
        :return:
        '''
        for i in range(reps):
            _, dont_save = divmod(i, 10)
            if not dont_save:
                self.save_checkpoint(i)
                print("done with round: " + str(i))

                self.train_gn(self.batch_size, check_acc=True)
                self.train_dn(i, self.batch_size, check_acc=True)

            self.train_dn(i, size=self.batch_size)
            self.train_gn(size=self.batch_size)

            #op so nice we'll do it twice
            self.train_gn(size=self.batch_size)



    def generate(self, size=100):
        '''
        Generates sample data from noise via the generator network.

        :param size:
        :return:
        '''
        feed_noise = np.random.uniform(-1, 1, size=[size, 100]).astype(np.float32)
        return self.sess.run(self.gen_y_sig, feed_dict={self.noise: feed_noise})


    def save_checkpoint(self, reps, save_state=True):
        fp_samples = self.generate(self.batch_size)
        rescaled_samples = fpdata.np_rescale(fp_samples, snap=False)
        sample_to_out = rescaled_samples[0]
        self.metmgr.update_all(reps, fp_samples)


        #uncomment below to get printouts of sample
        if self.debug == True:
            print(sample_to_out)

        #save parameter state
        if save_state:
            self.saver.save(self.sess, "./checkpoints/state" + str(reps) + ".ckpt")

        self.datamgr.import_sample_fp(sample_to_out)

        self.datamgr.export_svg(-1, "./samples/" + self.sample_label + "_" + str(reps) + ".svg")

