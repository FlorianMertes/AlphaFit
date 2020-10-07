import tensorflow as tf
import numpy as np

tf_type = tf.float32
np_type =  np.float32

class Nuclide():


    def __init__(self,emission_E,emission_I,num_tails=3):
        self.emission_energy = tf.constant(emission_E.astype(np_type).reshape(-1,1),dtype=tf_type)
        self.emission_intensity = tf.constant(emission_I.astype(np_type).reshape(-1,1),dtype=tf_type)
        self.num_components = tf.constant(self.emission_energy.shape[0])
        self.num_tails = tf.constant(num_tails)

        '''
        Fit variables 
        '''
        #self.taus = tf.Variable(tf.zeros(shape=(num_tails,1,1)),dtype=tf_type)
        self.taus = tf.Variable(np.array([1.,10.,100.]).reshape(-1,1,1),dtype=tf_type)
        self.sigma = tf.Variable([[[10.]]],dtype=tf_type)
        self.area = tf.Variable(100.,dtype=tf_type)
        #self._weights = tf.Variable(tf.zeros(shape=num_tails-1),dtype=tf_type)
        self._weights = tf.Variable(np.array([0.33,0.33]).astype(np_type),dtype=tf_type)

        self.recp_sqrt2 = tf.constant(np.array(1/np.sqrt(2.)).astype(np_type),dtype=tf_type)

    @tf.function()
    def evaluate(self,x):
        # this is some weird method to calculate the response, based on tensor shapes and inherent broadcasting
        w = tf.reshape(tf.concat([tf.expand_dims(1-tf.reduce_sum(self._weights),0),self._weights],axis=0),(-1,1,1))
        dist = x - self.emission_energy # results in 2d tensor
        sigtau = self.sigma / self.taus
        resp = w/(2*self.taus) * tf.math.exp(dist/self.taus + 0.5*sigtau*sigtau ) * tf.math.erfc(self.recp_sqrt2*( dist / self.sigma + sigtau))
        resp = tf.where(tf.math.is_nan(resp),0.,resp)
        res_sums = self.emission_intensity * tf.reduce_sum(resp,axis=0)
        return self.area * tf.reduce_sum(res_sums,axis=0)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    emissions = np.array([4563.,5200.])
    intensities = np.array([0.5,0.6])
    x = tf.cast(tf.range(4000,5500),tf_type)
    i = Nuclide(emissions,intensities)
    res = i.evaluate(x)
    plt.figure()
    for k in range(3):
        for i in range(2):
            plt.plot(res[k][i])
    #plt.yscale('log')
    plt.show()