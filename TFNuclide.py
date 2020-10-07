import tensorflow as tf
import numpy as np

tf_type = tf.float64
np_type =  np.float64

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
        self.area = tf.Variable(1000000.,dtype=tf_type)
        #self._weights = tf.Variable(tf.zeros(shape=num_tails-1),dtype=tf_type)
        self._weights = tf.Variable(np.array([0.33,0.33]).astype(np_type),dtype=tf_type)

        self.recp_sqrt2 = tf.constant(np.array(1/np.sqrt(2.)).astype(np_type),dtype=tf_type)
        self.zero = tf.constant(np.array(0.).astype(np_type),dtype=tf_type)
        self.thresh = tf.constant(np.array(300.).astype(np_type),dtype=tf_type)
        self.one = tf.constant(np.array(1.).astype(np_type),dtype=tf_type)
        self.thresh_arg1 = tf.constant(np.array(15.).astype(np_type),dtype=tf_type)
        self.thres_arg2 = tf.constant(np.array(10000.).astype(np_type),dtype=tf_type)


        self.trainable_variables = [self.area,self.taus,self.sigma,self._weights]

    @tf.function()
    def __call__(self,x):
        # this is some weird method to calculate the response, based on tensor shapes and inherent broadcasting
        w = tf.reshape(tf.concat([tf.expand_dims(1-tf.reduce_sum(self._weights),0),self._weights],axis=0),(-1,1,1))
        dist = (x-self.emission_energy)
        sigtau = self.sigma/self.taus
        '''
        Essentially, this will clip the exponential function in the model to avoid overflows.
        We replace the input argument with some unused dummy variables such that the backward gradient computations work and will
        then discard the dummy values.
        That way, we can ensure that the backward gradient computation does not produce NaNs in case of exponential function overflows
        '''
        disttau = (dist)/self.taus # results in 2d tensor
        distsig = (dist)/self.sigma
        arg1 = disttau + 0.5*sigtau*sigtau # compute all possible inputs to the exponential
        arg1_ok = arg1 < self.thresh_arg1 # get boolean array where clipping will occur
        arg2 = distsig + sigtau # same for error function input
        arg2_ok = arg2 < self.thres_arg2
        arg1 = tf.where(arg1_ok,arg1,self.one) # replace too high values with safe
        arg2 = tf.where(arg2_ok,arg2,self.one) # replace too high values with safe
        mask = tf.math.logical_and(arg1_ok,arg2_ok) # combine masks
        resp = tf.where(mask,w/(2*self.taus) * tf.math.exp(arg1) * tf.math.erfc(arg2),self.zero) # set to zero where clipping
        # resp is a a tensor of shape
        # (num_tails, num_peaks, len(x))
        # we use reductions to sum the values for each x
        res_sums = self.emission_intensity * tf.reduce_sum(resp,axis=0)
        return self.area * tf.reduce_sum(res_sums,axis=0)


class AdditiveModel():

    def __init__(self,models):
        self.models = []
        self.trainable_variables = []
        for k in models:
            self.trainable_variables += k.trainable_variables
            self.models.append(k)
        self.num_nuclides = len(self.models)

    def add_nuclide(self,nucl):
        self.models.append(nucl)
        self.num_nuclides += 1
        self.trainable_variables += nucl.trainable_variables

    @staticmethod
    @tf.function()
    def evaluate(x,models):
        print('tracing')
        resp = tf.zeros_like(x)
        for i in models:
            resp += i(x)
        return resp

    def __call__(self,x):
        return AdditiveModel.evaluate(x,self.models)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    emissions = np.array([4563.,5200.])
    intensities = np.array([0.5,0.6])
    x = tf.cast(tf.range(4000,5500),tf_type)
    i = Nuclide(emissions,intensities)
    res = i.evaluate(x)
    plt.figure()
    plt.plot(res)
    #plt.yscale('log')
    plt.show()