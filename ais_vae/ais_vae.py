from ais import *
from vae import *


# class AIS_VAE : 可使用AIS计算边缘概率密度的VAE
#
# 实例化参数： x_dim------------观测数据维度---------------------type : tfp.distributions.distribution 
#             z_dim------------隐变量维度-----------------------type : callable python function 
#             encoder_layers---编码器各层神经元数量--------------type : list
#             decoder_layers---解码器各层神经元数量--------------type : list 
#             num_chains-------独立的AIS链数量(并行采样)---------type : int
#             num_steps--------AIS链MCMC转移步数----------------type : int
#
# 接口函数：   AIS_log_prob( x)
#             函数功能 ： 通过AIS计算给定观测数据x的边缘概率密度
#             输入 ：观测数据x
#             输出 : x的对数边缘概率密度logp(x) 
#
#             train
#             函数功能 ： 训练AIS_VAE
#             输入 ：观测数据x
#             输出 : 无 
#
#             KDE_log_prob
#             函数功能 ： 通过KDE方法计算边缘概率密度
#             输入 ：观测数据x、KDE采样数量n_samples
#             输出 : 边缘概率密度 
#
#             get_ELBO
#             函数功能 ： 获取变分下界
#             输入 ：观测数据x
#             输出 : 变分下界 

class AIS_VAE( VAE):
    def __init__( self, x_dim, z_dim, encoder_layers, decoder_layers,
                    num_chains, num_steps, dtype=np.float64, sess=tf.Session()):
        super().__init__( x_dim, z_dim, encoder_layers, decoder_layers, dtype, sess)

        def log_prob_zx( z_, x_):
            y_out = self.batch_mlp( z_, self.decoder_layers, variable_scope='vae_graph/decoder')
            mn, sd = tf.split( y_out, num_or_size_splits=2, axis=-1)
            sd = tf.sigmoid( sd)
            p_z_x = tfd.MultivariateNormalDiag( mn, sd)
            return p_z_x.log_prob( x_) + self.pri_dist.log_prob( z_)

        self.ais = AIS_Bayes( self.pri_dist, log_prob_zx, num_chains, num_steps, x_dim, dtype, sess)

        self.sess.run( tf.global_variables_initializer())

    def AIS_log_prob( self, x):
        ais_prob = []
        for x_ in x:
            ais_prob.append( self.ais.get_marginal_prob( x_))
        return np.array( ais_prob)