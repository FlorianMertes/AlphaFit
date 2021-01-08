#ifndef TAIL_H

#define TAIL_H


typedef struct TailStruct
{
    int sigma_index;
    int tau_index;
    int mu_index;
    double* T;
    double* J;
    double* dTdtheta;
    int xlen;
} c_Tail;

void evaluate_T(c_Tail*,double*,double*);
void evaluate_T_and_J(c_Tail*, double*, double*);
void evaluate_tail_jac(c_Tail*, double*, double*);
void evaluate_tail_hess(c_Tail*, double*, double*);

#endif