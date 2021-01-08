/*
Tail Structure:
holds indices for its variables, which index a parameter location in a given array for its parameters
sigma
tau
mu

then, the mathematical definition of the Tail-function is

T = 1/(2*theta) * exp((x - mu)/tau + sigma²/2tau²) * erfc(((x-mu)/sigma + sigma/tau)/sqrt(2))

which is inherently numerically unstable for certain arguments of the exponential term.
This issue can be alleviated using the scaled complementary error function erfcx and rewrite T in terms of this function. We use the implementation in the
"Faddeeva" package to compute the erfc and erfcx terms.

dTdtheta are the derivatives of T wrt. sigma, tau, mu in that order, which are easily calculated using J

J = 1/(sqrt(2 pi)tau) * exp(-(x-mu)²/2sigma²)

*/

#include <math.h>
#include "Faddeeva.h"
#include "Tail.h"

#define M_1_SQRT2PI M_SQRT1_2/M_PI



void inline evaluate_T(c_Tail* tail, double* x, double* p)
{
    double distance = (*x) - p[tail->mu_index];
    double sig_tau = p[tail->sigma_index]/p[tail->tau_index];
    double a1 = distance/p[tail->tau_index] + 0.5*sig_tau*sig_tau;
    double a2 = M_SQRT1_2*(distance/p[tail->sigma_index] + sig_tau);
    if(a1 > 0.)
    {
        (*(tail->T)) = 0.5*p[tail->tau_index] * exp(-0.5*distance*distance/p[tail->sigma_index]/p[tail->sigma_index]) * Faddeeva_erfcx_re(a2);
    }
    else
    {
        (*(tail->T)) = 0.5*p[tail->tau_index] * exp(a1) * Faddeeva_erfc_re(a2);
    }
}

void inline evaluate_T_and_J(c_Tail* tail, double* x, double* p)
{
    double distance = *x - p[tail->mu_index];
    double sig_tau = p[tail->sigma_index]/p[tail->tau_index];
    double a1 = distance/p[tail->tau_index] + 0.5*sig_tau*sig_tau;
    double a2 = M_SQRT1_2*(distance/p[tail->sigma_index] + sig_tau);
    double a3 = -0.5*distance*distance/p[tail->sigma_index]/p[tail->sigma_index];
    if(a1 > 0.)
    {
        *(tail->T) = 0.5*p[tail->tau_index] * exp(a3) * Faddeeva_erfcx_re(a2);
    }
    else
    {
        *(tail->T) = 0.5*p[tail->tau_index] * exp(a1) * Faddeeva_erfc_re(a2);   
    }
    *(tail->J) = M_1_SQRT2PI/p[tail->tau_index] * exp(a3);
}

void evaluate_tail_jac(c_Tail* tail, double* x, double* p)
{
    evaluate_T_and_J(tail,x,p);
    double distance = *x - p[tail->mu_index];
    double sigtau = p[tail->sigma_index]/p[tail->tau_index];
    tail->dTdtheta[0] = -(1.d)*(*(tail->J))*(1.d/p[tail->tau_index] - distance/p[tail->sigma_index]/p[tail->sigma_index]); 
    tail->dTdtheta[1] = *(tail->J)*sigtau/p[tail->tau_index] - (*(tail->T))/p[tail->tau_index]*(sigtau*sigtau/p[tail->sigma_index] + distance/p[tail->tau_index]/p[tail->tau_index] - 1.d);
    tail->dTdtheta[2] = (*(tail->J))/p[tail->sigma_index] + (*(tail->T))/p[tail->tau_index];
}

void evaluate_tail_hess(c_Tail* tail, double* x, double* p)
{

}