import sympy as sp
import numpy as np
import numba as nb
import math
from sympy.printing import pycode

'''
@nb.jit('{sign_inpt}',{jit_flags})
'''


def generate_numba_code(variables,exprs,constants={},func_name="generated_func",jit_flags="nopython=True,nogil=True,parallel=True",verbose=True):
    subs_exprs = [e.subs(constants) for e in exprs]
    inpt_params = ','.join(f'{v}' for v in variables)
    terms_cse, return_expr = sp.cse(subs_exprs)
    if len(exprs) == 1:
        return_type = 'float64'
        return_code = pycode(return_expr[0])
    else:
        return_type = 'float64[:]'
        return_code = 'np.array('+pycode(return_expr)+',dtype=np.float64)'
    
    code_terms = []
    for e in terms_cse:
        k,v = e
        code_terms.append(f'{k}={pycode(v)}')
    code_terms = '\n\t'.join(code_terms)
    
    sign_inpt = '(' + ','.join('float64' for i in variables)+')'

    sign_inpt = return_type+sign_inpt

    template = f"""
@nb.jit('{sign_inpt}',{jit_flags})
def {func_name}({inpt_params}):
\t{code_terms}
\treturn {return_code}
"""
    if verbose:
        print("Attempting to JIT compile generated function: ")
        print(template)
    exec(template,globals(),locals()) # this compiles the function and adds to local dict
    return locals()[func_name] # returns respective functions from local dict



class Nuclide():

    def __init__(self,emission_energies,emission_intensities,num_tails=3,shared_sigma=True,nuclide_name="BlankName",fix_energies=True,fix_intensities=True):
        self.emission_energies = emission_energies
        self.emission_intensities = emission_intensities
        self.num_tails = num_tails
        self.shared_sigma = shared_sigma
        self.name = nuclide_name
        self.fix_energies = True
        self.fix_intensities = True
        self._construct_symbolic_equations_shared_sigma()
        self._jit_compile_funcs()


    def _construct_symbolic_equations_shared_sigma(self):
        '''
        Method to construct the symbolic fitting equations through SymPy and add the respective members
        '''
        '''
        TODO: Think about a better way to do this and possibly avoid the ugly control flow (nested control flow statements)
        '''
        self._sym_taus = sp.Array(sp.symbols(','.join(['tau%s'%(i) for i in range(0,self.num_tails)])))
        self._sym_area = sp.Symbol('A')
        self._sym_sigma = sp.Symbol('sigma')
        temp_sym_weights = sp.Array(sp.symbols(['w%s'%(i) for i in range(1,self.num_tails)]))
        n = sp.Symbol('n')
        self._sym_weights = sp.Array([1-sp.Sum(temp_sym_weights[n],(n,0,self.num_tails-2)).doit(),*temp_sym_weights])
        self._sym_x = sp.Symbol('x')
        mu = sp.Symbol('mu')
        per_peak_expr = sp.Sum((self._sym_weights[n]/(2*self._sym_taus[n]))*sp.exp((self._sym_x-mu)/(self._sym_taus[n])+self._sym_sigma**2/(2*self._sym_taus[n]**2))*sp.erfc((1/sp.sqrt(2))*((self._sym_x-mu)/self._sym_sigma + self._sym_sigma/self._sym_taus[n])),(n,0,self.num_tails-1))
        self._per_peak_expr = per_peak_expr.doit()
        if len(self.emission_energies) == 1:
            self._sym_center = mu
            self._sym_intensities = sp.Symbol('p')
            self._sym_expr = self._sym_area*self._sym_intensities*self._per_peak_expr
        else:
            self._sym_center = sp.Array(sp.symbols(','.join(['mu%s'%(i) for i in range(len(self.emission_energies))])))
            self._sym_intensities = sp.Array(sp.symbols(','.join(['p%s'%(i) for i in range(len(self.emission_energies))])))
            self._sym_expr = self._sym_area*sp.Sum(self._sym_intensities[n]*self._per_peak_expr.subs({mu:self._sym_center[n]}),(n,0,len(self.emission_energies)-1)).doit()
        self.parameters = dict()
        self.constants = dict()
        for k in [self._sym_area,self._sym_taus,self._sym_sigma,self._sym_weights]:
            for s in k.free_symbols:
                self.parameters[s] = 0.
        if not self.fix_energies:
            i = 0
            for s in self._sym_center.free_symbols:
                self.parameters[s] = self.emission_energies[i]
                i += 1
        else:
            i = 0
            for s in self._sym_center.free_symbols:
                self.constants[s] = self.emission_energies[i]
                i += 1
        if not self.fix_intensities:
            i = 0
            for s in self._sym_intensities.free_symbols:
                self.parameters[s] = self.emission_intensities[i]
                i += 1
        else:
            i = 0
            for s in self._sym_intensities.free_symbols:
                self.constants[s] = self.emission_intensities[i]
                i += 1
        self._sym_deriv_expr = [sp.diff(self._sym_expr,p) for p in self.parameters]

    def _jit_compile_funcs(self):
        inpt_sign = [self._sym_x] + list(self.parameters.keys())
        code = generate_numba_code(inpt_sign,[self._sym_expr],self.constants)
        setattr(self,'fit_func',code)
        #code = generate_numba_code(inpt_sign,self._sym_deriv_expr,self.constants)
        #setattr(self,'fit_jac',code)
        _p = dict()
        _c = dict()
        for k in self.parameters.keys():
            _p[k.name] = self.parameters[k]
        self.parameters = _p
        for k in self.constants.keys():
            _c[k.name] = self.constants[k]
        self.constants = _c

    def __call__(self,x):
        y = np.zeros_like(x)
        for k in range(len(x)):
            y[k] = self.fit_func(x[k],*self.parameters.values())
        return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    energies = [4536.,5123.]
    intensities = [0.5,1.]
    f = Nuclide(energies,intensities)
    f.parameters['A'] = 100.
    f.parameters['w2'] = 0.33
    f.parameters['w1'] = 0.33
    f.parameters['tau0'] = 1.
    f.parameters['tau1'] = 100.
    f.parameters['tau2'] = 1000.
    f.parameters['sigma'] = 10.
    energy = np.arange(4000,6000).astype(np.float64)
    y = f(energy)
    plt.figure()
    plt.plot(energy,y)
    plt.show()