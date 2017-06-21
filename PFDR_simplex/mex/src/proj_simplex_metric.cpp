/*==================================================================
 * Hugo Raguet 2016
 *================================================================*/
#include <alloca.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>

/* constants of the correct type */
#define ZERO    ((real) 0.)
#define TRUE    ((uint8_t) 1)
#define FALSE   ((uint8_t) 0)

/* minimum problem size each thread should take care of within parallel regions */
#define CHUNKSIZE 1000

template <typename real>
void proj_simplex_metric(real *X, const real *M, const int D, const int N, \
                                    const int nm, const real *A, const int na)
{
    int d, n;
    real la, s, *x;
    const real *m;
    uint8_t c, *I; /* boolean table indicating coordinates greater than la */

    /* number of threads to use */
    const int nt = (D*N > omp_get_num_procs()*CHUNKSIZE) ? \
                    omp_get_num_procs() : 1 + (D*N  - CHUNKSIZE)/CHUNKSIZE;

    #pragma omp parallel private(d, n, la, s, x, m, c, I) num_threads(nt)
    {
    I = (uint8_t*) alloca(D*sizeof(uint8_t));
    #pragma omp for
    for (n = 0; n < N; n++){
        x = X + D*n;
        m = (nm > n) ? M + D*n : M + D*(nm - 1);
        la = (na > n) ? (x[0] - A[n])/m[0] : (x[0] - A[na - 1])/m[0];
        x[0] = x[0]/m[0];
        I[0] = TRUE;
        s = m[0];
        /* first pass: populate I and x */
        for (d = 1; d < D; d++){
            x[d] = x[d]/m[d];
            if (x[d] > la){
                I[d] = TRUE;
                s += m[d];
                la += m[d]*(x[d] - la)/s;
            }else{
                I[d] = FALSE;
            }
        }
        /* subsequent passes */
        c = TRUE;
        while (c){
            c = FALSE;
            for (d = 0; d < D; d++){
                if (I[d]){
                    if (x[d] < la){
                        I[d] = FALSE;
                        s -= m[d];
                        la += m[d]*(la - x[d])/s;
                        c = TRUE;
                    }
                }
            }
        }
        /* finalize */
        for (d = 0; d < D; d++){
            if (I[d]){
                x[d] = (x[d] - la)*m[d];
            }else{
                x[d] = ZERO;
            }
        }
    }
    }
}

/* instantiate for compilation */
template void proj_simplex_metric<float>(float*, const float*, const int, \
                                const int, const int, const float*, const int);
template void proj_simplex_metric<double>(double*, const double*, const int, \
                                const int, const int, const double*, const int);
