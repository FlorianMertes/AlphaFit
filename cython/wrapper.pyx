#cython: nonecheck = False
#cython: cdivision = True
#cython: boundscheck = False
#cython: wraparound = False
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
import numpy as np
from cython.parallel import prange

cdef extern from "Tail.h":
    struct TailStruct:
        int sigma_index
        int tau_index
        int mu_index
        double* T
        double* J
        double* dTdtheta
        int xlen
    
    ctypedef TailStruct c_Tail

    cdef inline void evaluate_T(c_Tail* tail, double* x, double* p,int index) nogil
    cdef inline void evaluate_tail_jac(c_Tail* tail, double* x, double* p, int index) nogil


cdef class Singleton():
    _instances = {}

    @classmethod
    def instance(cls,*args,**kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args,**kwargs)
        return cls._instances[cls]

cdef class Parameter():

    cdef object name
    cdef public double init_value

    def __init__(self,name='Default Parameter',init_value=1.):
        self.name = name
        self.init_value = init_value

    def __repr__(self):
        return self.name

cdef class Parameters(Singleton):

    cdef list params

    def make_parameter(self,*args,**kwargs):
        if not self.params:
            self.params = []
        p = Parameter(*args,**kwargs)
        self.params.append(p)
        return p, len(self.params)-1

    def remove_parameter(self,param):
        self.params.remove(param)

    def get_parameter_index(self,param):
        if not self.params:
            self.params = []
        return self.params.index(param)

    def get_parameters(self):
        if not self.params:
            self.params = []
        return self.params

def list_parameters():
    return Parameters.instance().get_parameters()

def get_p0():
    return np.array([i.init_value for i in Parameters.instance().get_parameters()])


cdef class Tail():

    cdef c_Tail* c_struct
    cdef object name
    cdef Parameter _sigma
    cdef Parameter _tau
    cdef Parameter _mu
    cdef _xlen

    def __cinit__(self,name='Default Tail',xlen=1):
        cdef int sigma_index
        cdef int tau_index
        cdef int mu_index
        self._sigma,sigma_index  = Parameters.instance().make_parameter('sigma_'+name) # make initial parameters
        self._tau,tau_index = Parameters.instance().make_parameter('tau_'+name) # make initial parameters
        self._mu,mu_index = Parameters.instance().make_parameter('mu_'+name) # make initial parameters
        self._xlen = xlen
        self.c_struct = <c_Tail*>malloc(sizeof(c_Tail)) # malloc space for c structure (which also holds pointers to the computation data)
        self.c_struct.T = <double*>malloc(xlen*sizeof(double)) # malloc space for value
        self.c_struct.J = <double*>malloc(xlen*sizeof(double)) # malloc space for auxiliary variable
        self.c_struct.dTdtheta = <double*>malloc(xlen*3*sizeof(double)) # malloc space for derivative
        self.c_struct.sigma_index = sigma_index
        self.c_struct.tau_index = tau_index
        self.c_struct.mu_index = mu_index
        self.c_struct.xlen = xlen

    @property
    def xlen(self):
        return self._xlen

    @xlen.setter
    def xlen(self,int other):
        free(self.c_struct.T)
        free(self.c_struct.J)
        free(self.c_struct.dTdtheta)
        self.c_struct.dTdtheta = <double*>malloc(other*3*sizeof(double)) # reallocate space for different size of x
        self.c_struct.T = <double*>malloc(other*sizeof(double)) # reallocate space for different size of x
        self.c_struct.J = <double*>malloc(other*sizeof(double)) # reallocate space for different size of x
        self._xlen = other
        self.c_struct.xlen = other

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self,Parameter other):
        new_sigma_index = Parameters.instance().get_parameter_index(other)
        Parameters.instance().remove_parameter(self._sigma)
        self._sigma = other
        self.c_struct.sigma_index = new_sigma_index

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, Parameter other):
        new_tau_index = Parameters.instance().get_parameter_index(other)
        Parameters.instance().remove_parameter(self._tau)
        self._tau = other
        self.c_struct.tau_index = new_tau_index

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, Parameter other):
        new_mu_index = Parameters.instance().get_parameter_index(other)
        Parameters.instance().remove_parameter(self._mu)
        self._mu = other
        self.c_struct.mu_index = new_mu_index


cdef class Peak():

    cdef public list tails
    cdef int num_tails
    cdef c_Tail** tail_c_structs
    cdef int _xlen

    def __cinit__(self,name='Default Peak',num_tails=3,int xlen=1):
        self._xlen = xlen
        self.num_tails = num_tails
        self.tails = []
        self.tail_c_structs = <c_Tail**>malloc(num_tails*sizeof(c_Tail*))
        for i in range(num_tails):
            t = Tail(name=name+'_'+str(i),xlen=xlen)
            if i != 0:
                t.mu = self.tails[0].mu
            self.tails.append(t)
            self.tail_c_structs[i] = t.c_struct

    cpdef evaluate(self,double[:] x, double[:] p):
        for i in range(self.num_tails):
            if self._xlen != self.tail_c_structs[i].xlen:
                pass

    @property
    def sigma(self):
        return [i.sigma for i in self.tails]

    @sigma.setter
    def sigma(self,list_of_sigmas):
        for i,k in enumerate(list_of_sigmas):
            self.tails[i].sigma = k

    @property
    def tau(self):
        return [i.tau for i in self.tails]

    @tau.setter
    def tau(self,list_of_taus):
        for i,k in enumerate(list_of_taus):
            self.tails[i].tau = k

    @property
    def mu(self):
        return [i.mu for i in self.tails]

    @mu.setter
    def mu(self,list_of_mus):
        for i,k in enumerate(list_of_mus):
            self.tails[i].mu = k