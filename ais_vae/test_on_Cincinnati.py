from ais_vae import *
import matplotlib.pyplot as plt
import scipy.io as scipy_io


x = np.load( './x_fft.npy')
x = np.mean( x, axis=1)

with tf.Session() as sess:
    ais_vae = AIS_VAE( x_dim=301, z_dim=3, num_chains=1000, num_steps = 100,
                         encoder_layers=[128 , 64 , 3*2], decoder_layers=[32, 64, 301*2], sess=sess)
  
    ais_vae.load( './model/model')

    ais_data = ais_vae.AIS_log_prob( x)
    kde_data = ais_vae.KDE_log_prob( x, 1000)
    elbo_data = ais_vae.get_ELBO( x)

    data = { 'ais_data' : ais_data,
             'kde_data' : kde_data,
             'elbo_data' : elbo_data} 

    scipy_io.savemat('./test_result.mat', data)



    # for i in range( 100000):
    #     sample_x = x[ np.random.choice( len(x[:100]), size=64, replace=False)] 
    #     ais_vae.train( sample_x)
    #     if not i % 200:
    #         elbo = ais_vae.get_ELBO( x)
    #         eobo = np.clip( elbo, a_min=500, a_max=1500)
    #         plt.plot( np.arange( 0., 1., 1./len( elbo)), elbo)
    #         plt.show()