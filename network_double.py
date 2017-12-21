import tensorflow as tf
import numpy as np

class network_simple(object):

    def __init__(self,session,resolution,n_action, learning_rate):

        self.resolution = resolution

        self.s1_ = tf.placeholder(tf.float32, [None] + [resolution[0],resolution[1], resolution[2]], name="State")
        self.a_ = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q_ = tf.placeholder(tf.float32, [None, n_action], name="TargetQ")
        self.reward_in = tf.placeholder(tf.float32, name="reward")
        self.reward_out = tf.identity(self.reward_in)
        self.total_reward_ = tf.placeholder(tf.float32, name="total_reward")

        with tf.name_scope("conv1"):
            self.conv1_weight = tf.Variable(tf.truncated_normal(shape=[6,6,resolution[2],8],stddev=0.1),name="weight")
            self.conv1_strides = [1,3,3,1]
            self.conv1_padding = "SAME"
            self.conv1_bias = tf.Variable(tf.constant(0.1, shape=[8]),name="bias")

            self.conv1 = tf.nn.conv2d(self.s1_,self.conv1_weight,self.conv1_strides,self.conv1_padding,name="conv")
            self.conv1_relu = tf.nn.relu(tf.nn.bias_add(self.conv1, self.conv1_bias),name="relu") 

        with tf.name_scope("conv1_target"):
            self.conv1_weight_target = tf.Variable(tf.constant(0.0,shape=[6,6,resolution[2],8]), name='weight_target')
            self.conv1_bias_target = tf.Variable(tf.constant(0.0,shape=[8]), name='bias_target')

            self.conv1_target = tf.nn.conv2d(self.s1_,self.conv1_weight_target,self.conv1_strides,self.conv1_padding,name="conv1_target")
            self.conv1_relu_target = tf.nn.relu(tf.nn.bias_add(self.conv1_target, self.conv1_bias_target), name="relu_target")

        with tf.name_scope("conv2"):
            self.conv2_weight = tf.Variable(tf.truncated_normal(shape=[3,3,8,8],stddev=0.1),name="weight")
            self.conv2_strides = [1,2,2,1]
            self.conv2_padding = "SAME"
            self.conv2_bias = tf.Variable(tf.constant(0.1,shape=[8]))

            self.conv2 = tf.nn.conv2d(self.conv1_relu, self.conv2_weight, self.conv2_strides,self.conv2_padding,name="conv")
            self.conv2_relu = tf.nn.relu(tf.nn.bias_add(self.conv2,self.conv2_bias),name="relu")

        with tf.name_scope("conv2_target"):
            self.conv2_weight_target = tf.Variable(tf.constant(0.0,shape=[3,3,8,8]), name='weight_target')
            self.conv2_bias_target = tf.Variable(tf.constant(0.0,shape=[8]), name='bias_target')

            self.conv2_target = tf.nn.conv2d(self.conv1_relu_target,self.conv2_weight_target,self.conv2_strides,self.conv2_padding,name="conv2_target")
            self.conv2_relu_target = tf.nn.relu(tf.nn.bias_add(self.conv2_target, self.conv2_bias_target), name="relu_target")

        with tf.name_scope("reshape"):
            self.reshape = tf.contrib.layers.flatten(self.conv2_relu)

        with tf.name_scope("reshape_target"):
            self.reshape_target = tf.contrib.layers.flatten(self.conv2_relu_target)

        with tf.name_scope("fc1"):
            self.fc1_weights = tf.Variable(tf.truncated_normal(shape=[self.reshape.get_shape()[1].value,128],stddev=0.1),name="weight")
            self.fc1_bias = tf.Variable(tf.constant(0,1,shape=[128]),name="bias")
            self.fc1 = tf.nn.bias_add(tf.matmul(self.reshape,self.fc1_weights),self.fc1_bias,name="fc")
            self.fc1_relu = tf.nn.relu(self.fc1,name="relu")

        with tf.name_scope("fc1_target"):
            self.fc1_weight_target = tf.Variable(tf.constant(0.0,shape=[self.reshape_target.get_shape()[1].value, 128]), name='weight_target')
            self.fc1_bias_target = tf.Variable(tf.constant(0.0,shape=[128]), name='bias_target')

            self.fc1_target = tf.nn.bias_add(tf.matmul(self.reshape_target, self.fc1_weight_target),self.fc1_bias_target,name="fc1_target")
            self.fc1_relu_target = tf.nn.relu(self.fc1_target, name="relu_target")

        with tf.name_scope("fc2"):
            self.fc2_weights = tf.Variable(tf.truncated_normal(shape=[128,n_action],stddev=0.1),name="weight")
            self.fc2_bias = tf.Variable(tf.constant(0,1,shape=[n_action]),name="bias")
            self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1_relu, self.fc2_weights),self.fc2_bias,name="fc")
            self.q = self.fc2
            #self.q = tf.nn.relu(self.fc2,name="relu")

        with tf.name_scope("fc2_target"):
            self.fc2_weight_target = tf.Variable(tf.constant(0.0,shape=[128, n_action]), name='weight_target')
            self.fc2_bias_target = tf.Variable(tf.constant(0.0,shape=[n_action]), name='bias_target')

            self.fc2_target = tf.nn.bias_add(tf.matmul(self.fc1_relu_target, self.fc2_weight_target),self.fc2_bias_target,name="fc2_target")
            self.q_target = self.fc2_target
            #self.fc2_relu_target = tf.nn.relu(self.fc2_target, name="relu_target")
        
        self.print_shapes()

        self.best_action = tf.argmax(self.q,1)

        self.loss = tf.losses.mean_squared_error(self.q,self.target_q_)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        self.session = session

        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

        self.copy_params()

        self.saver = tf.train.Saver()

        with tf.name_scope("summary"):
            tf.summary.image('s1_',tf.reshape(self.s1_,[-1]+list(self.s1_.get_shape()[1:])),1)
            tf.summary.image('conv1',tf.reshape(tf.transpose(self.conv1_relu,perm=[0,3,1,2]),[-1]+list(self.conv1_relu.get_shape()[1:3])+[1]),1)
            tf.summary.image('conv2',tf.reshape(tf.transpose(self.conv2_relu,perm=[0,3,1,2]),[-1]+list(self.conv2_relu.get_shape()[1:3])+[1]),1)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reward', self.reward_out)
            self.merged = tf.summary.merge_all()
            self.total_reward_graph = tf.summary.scalar('total_reward', self.total_reward_)
            self.writer = tf.summary.FileWriter("./logs", self.session.graph)

    def learn(self, s1, target, reward, step):
        l, _, m = self.session.run([self.loss, self.train_step,self.merged], feed_dict={self.s1_:s1, self.target_q_:target, self.reward_in:reward})
        
        if step %10 == 0:
            self.writer.add_summary(m,step)
        return l

    def get_q_values(self, state):
        return self.session.run(self.q, feed_dict={self.s1_:state})

    def get_q_target_values(self,state):
        best_actions = self.get_best_actions(state)
        q =  self.session.run(self.q_target, feed_dict={self.s1_:state})
        ret = []
        for i in range(len(q)):
            ret.append(q[i][best_actions[i]])
        ret = np.array(ret)
        return ret

    def get_best_actions(self,state):
        s1 = state.reshape([-1,self.resolution[0],self.resolution[1],self.resolution[2]])
        return self.session.run(self.best_action, feed_dict={self.s1_:s1})

    def get_best_action(self,state):
        s1 = state.reshape([-1,self.resolution[0],self.resolution[1],self.resolution[2]])
        return self.session.run(self.best_action, feed_dict={self.s1_:s1})[0]

    def write(self, s1):
        self.writer.add_summary(self.session.run(self.merged,feed_dict={self.s1_:s1}))
    
    def write_total_reward(self, total_reward, step):
        self.writer.add_summary(self.session.run(self.total_reward_graph,feed_dict={self.total_reward_:total_reward}),step)

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self,model_path):
        self.saver.restore(self.session,model_path)

    def copy_params(self):
        origin_params = [self.conv1_weight, self.conv1_bias,self.conv2_weight, self.conv2_bias, self.fc1_weights,self.fc1_bias,self.fc2_weights,self.fc2_bias]
        target_params = [self.conv1_weight_target, self.conv1_bias_target, self.conv2_weight_target, self.conv2_bias_target, self.fc1_weight_target, self.fc1_bias_target, self.fc2_weight_target, self.fc2_bias_target]

        self.copyop = [tf.assign(target, origin) for origin,target in zip(origin_params,target_params) ]
        self.session.run(self.copyop)

    def print_shapes(self):
        print(self.s1_.name,"-------")
        print(self.s1_.get_shape(),"\n")

        print(self.conv1_relu.name,"-------")
        print(self.conv1_relu.get_shape(),"\n")

        print(self.conv2_relu.name,"-------")
        print(self.conv2_relu.get_shape(),"\n")

        print(self.fc1_relu.name,"-------")
        print(self.fc1_relu.get_shape(),"\n")

        print(self.q.name,"-------")
        print(self.q.get_shape(),"\n")