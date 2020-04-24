import numpy as np
import tensorflow
import tensorflow_probability as tfp
tf = tensorflow.compat.v1 
tf.disable_v2_behavior()
tfd = tfp.distributions


# class AIS_Bayes : 适用于贝叶斯概率推断下的AIS采样计算类
#
# 实例化参数： pz-------------先验分布p(z)---------------------type : tfp.distributions.distribution 
#             log_prob_zx----联合分布对数密度函数logp(z,x)-----type : callable python function 
#             num_chains-----独立的AIS链数量(并行采样)---------type : int
#             num_steps------AIS链MCMC转移步数----------------type : int 
#             x_dims---------观测数据x的维度------------------type : int
#
# 接口函数：   get_marginal_prob( x)
#             函数功能 ： 通过AIS计算给定观测数据x的边缘概率密度
#             输入 ：观测数据x
#             输出 : x的对数边缘概率密度logp(x) 

class AIS_Bayes:
    def __init__( self, pz, log_prob_zx, num_chains, num_steps, x_dims, dtype=np.float32, sess=tf.Session()):
        self.x_op = tf.placeholder( shape=(x_dims,), dtype=dtype)  
        self.x_op = tf.tile( (self.x_op,), multiples=[num_chains, 1])

        self.num_chains = num_chains

        self.prior_dist = pz   
        self.prior_dist.ex_log_prob = lambda z : pz.log_prob( z)
        # self.joint_dist_log_prob_fn = lambda z : log_prob_zx( 
        #                                             tf.concat( [z, tf.tile( (self.x_op,), multiples=[num_chains, 1])], axis=1))
        self.joint_dist_log_prob_fn = lambda z : log_prob_zx( z, self.x_op)

        self.weight_samples, self.ais_weights, self.kernel_results = (
                tfp.mcmc.sample_annealed_importance_chain(
                        num_steps = num_steps,
                        proposal_log_prob_fn = self.prior_dist.ex_log_prob,
                        target_log_prob_fn = self.joint_dist_log_prob_fn,
                        current_state = self.prior_dist.sample( num_chains),
                        make_kernel_fn = lambda tlp_fn : tfp.mcmc.HamiltonianMonteCarlo(
                                target_log_prob_fn = tlp_fn,
                                step_size = 0.1,
                                num_leapfrog_steps = 2),
                        parallel_iterations = 10))

        self.marginal_prob_op = tf.reduce_logsumexp( self.ais_weights) - np.log( num_chains)

        self.sess = sess
        self.sess.run( tf.global_variables_initializer())

    def get_marginal_prob( self, x):
        x = np.tile( [x],  [self.num_chains, 1])
        return self.sess.run( self.marginal_prob_op, feed_dict = { self.x_op : x })


