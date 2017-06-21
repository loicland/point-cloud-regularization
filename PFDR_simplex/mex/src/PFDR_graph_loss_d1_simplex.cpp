/*==================================================================
 * Hugo Raguet 2016
 *================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <cmath> /* for log function, only for computing KLa loss */
#ifdef _OPENMP
    #include <omp.h>
#endif
#ifdef MEX
    #include "mex.h"
    #define FLUSH mexEvalString("drawnow expose")
#else
    #define FLUSH fflush(stdout)
#endif
#include "../include/proj_simplex.hpp"
#include "../include/PFDR_graph_loss_d1_simplex.hpp"

/* constants of the correct type */
#define ZERO ((real) 0.)
#define ONE ((real) 1.)
#define TWO ((real) 2.)
#define HALF ((real) 0.5)
#define ALMOST_TWO ((real) 1.9)
#define TENTH ((real) 0.1)

/* minimum problem size each thread should take care of within parallel regions */
#define CHUNKSIZE 1000

static inline int compute_num_threads(const int size)
{
#ifdef _OPENMP
    if (size > omp_get_num_procs()*CHUNKSIZE){
        return omp_get_num_procs();
    }else{
        return 1 + (size - CHUNKSIZE)/CHUNKSIZE;
    }
#else
    return 1;
#endif
}

template<typename real>
static void print_progress(char *msg, int it, int itMax, const real dif, \
                                        const real difTol, const real difRcd)
{
    int k = 0;
    while (msg[k++] != '\0'){ printf("\b"); }
    sprintf(msg, "iteration %d (max. %d)\n", it, itMax);
    if (difTol > ZERO || difRcd > ZERO){
        if (difTol >= ONE){
            sprintf(msg, "%slabel evolution %d (recond. %d; tol. %d)\n", \
                                msg, (int) dif, (int) difRcd, (int) difTol);
        }else{
            sprintf(msg, "%siterate evolution %g (recond. %g; tol. %g)\n", \
                                                 msg, dif, difRcd, difTol);
        }
    }
    printf("%s", msg);
    FLUSH;
}

template<typename real>
static void preconditioning(const int K, const int V, const int E, \
    const real al, const real *P, const real *Q, \
    const int *Eu, const int *Ev, const real *La_d1, \
    real *Ga, real *GaQ, real *Zu, real *Zv, real *Wu, real *Wv, \
    real *W_d1u, real *W_d1v, real *Th_d1, const real rho, const real condMin)
/* 20 arguments 
 * for initialization:
 *      Zu, Zv are NULL
 * for reconditioning:
 *      Zu, Zv are the current auxiliary variables for d1 */
{
    /**  control the number of threads with Open MP  **/
    const int ntVK = compute_num_threads(V*K);
    const int ntEK = compute_num_threads(E*K);
    const int ntKE = (ntEK < K) ? ntEK : K;

    /**  initialize general variables  **/
    int u, v, e, k, i; /* index edges and vertices */
    real a, b, c; /* general purpose temporary real scalars */
    real *Aux; /* auxiliary pointer */
    real al_K, al_1, al_K_al_1;
    if (ZERO < al && al < ONE){ /* constants for KLa loss */
        al_K = al/K;
        al_1 = ONE - al;
        al_K_al_1 = al_K/al_1;
    }

    if (Zu != NULL){ /* reconditioning */
        /**  retrieve original metric 
         **  normalized after last preconditioning  **/
        if (al == ONE){ /* quadratic loss, GaQ is a backup for Ga */
            for (v = 0; v < V*K; v++){ Ga[v] = GaQ[v]; }
        }else if (al > ZERO){ /* KLa loss */
            #pragma omp parallel for private(v) num_threads(ntVK)
            for (v = 0; v < V*K; v++){ Ga[v] = GaQ[v]/(al_K + al_1*Q[v]); }
        }else{ /* linear loss, GaQ = Ga*Q, but some Q are zero */
            #pragma omp parallel for private(u, v, k, i, a) num_threads(ntVK)
            for (u = 0; u < V; u++){
                v = u*K;
                /* use highest value of Q to improve accuracy */
                i = 0;
                a = Q[v];
                for (k = 1; k < K; k++){
                    if (a < Q[v+k]){
                        a = Q[v+k];
                        i = k;
                    }
                }
                /* retrieve normalization coefficient */
                a = GaQ[v+i]/a/Ga[v+i];
                for (k = 0; k < K; k++){ Ga[v+k] *= a; } 
            }
        }
        /**  get the auxiliary subgradients  **/
        #pragma omp parallel for private(e, i, u, v, k) num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            for (k = 0; k < K; k++){ 
                if (al == ZERO){ /* linear loss, grad = -Q */
                    Zu[i] = (Wu[i]/Ga[u])*(P[u] + GaQ[u] - Zu[i]);
                    Zv[i] = (Wv[i]/Ga[v])*(P[v] + GaQ[v] - Zv[i]);
                }else if (al == ONE){ /* quadratic loss, grad = P - Q */
                    Zu[i] = (Wu[i]/Ga[u])*(P[u] - Ga[u]*(P[u] - Q[u]) - Zu[i]);
                    Zv[i] = (Wv[i]/Ga[v])*(P[v] - Ga[v]*(P[v] - Q[v]) - Zv[i]);
                }else{ /* dKLa/dp_k = -(a/K + (1-a)q_k)(1-a)/(a/K + (1-a)p_k) */
                    Zu[i] = (Wu[i]/Ga[u])*(P[u] + GaQ[u]/(al_K_al_1 + P[u]) - Zu[i]);
                    Zv[i] = (Wv[i]/Ga[v])*(P[v] + GaQ[v]/(al_K_al_1 + P[v]) - Zv[i]);
                }
                i++; u++; v++;
            }
        }
    }
    
    /**  compute the Hessian  **/
    if (al == ZERO){ /* linear loss, H = 0 */
        for (v = 0; v < V*K; v++){ Ga[v] = ZERO; }
    }else if (al == ONE){ /* quadratic loss, H = 1 */
        for (v = 0; v < V*K; v++){ Ga[v] = ONE; }
    }else{ /* d^2KLa/dp_k^2 = (a/K + (1-a)q_k)(1-a)^2/(a/K + (1-a)p_k)^2 */
        #pragma omp parallel for private(v, a) num_threads(ntVK)
        for (v = 0; v < V*K; v++){
            a = (al_K_al_1 + P[v]);
            Ga[v] = (al_K + al_1*Q[v])/(a*a);
        }
    }

    /**  d1 contribution and splitting weights  **/
    if (al == ZERO){ /* linear case, compute directly pseudo hessian */
        Aux = Ga;
    }else{ /* use GaQ as temporary storage */
        for (v = 0; v < V*K; v++){ GaQ[v] = ZERO; } 
        Aux = GaQ;
    }
    /* this task cannot be easily parallelized along the edges */
    #pragma omp parallel for private(k, e, u, v, i, a) num_threads(ntKE)
    for (k = 0; k < K; k++){
        i = k;
        for (e = 0; e < E; e++){
            u = Eu[e]*K + k;
            v = Ev[e]*K + k;
            if (Zu == NULL){ /* first preconditioning */
                a = La_d1[e];
            }else{ /* reconditioning */
                a = P[u] - P[v];
                if (a < ZERO){ a = -a; }
                if (a < condMin){ a = condMin; }
                a = La_d1[e]/a;
            }
            Aux[u] += a;
            Aux[v] += a;
            Wu[i] = a;
            Wv[i] = a;
            i += K;
        }
    }
    if (al > ZERO){ /* add contribution to the Hessian */
        #pragma omp parallel for private(v) num_threads(ntVK)
        for (v = 0; v < V*K; v++){ Ga[v] += Aux[v]; }
    }
    /* inverse the sum of the weights */
    #pragma omp parallel for private(v) num_threads(ntVK)
    for (v = 0; v < V*K; v++){ Aux[v] = ONE/Aux[v]; }
    /* make splitting weights sum to unity */
    #pragma omp parallel for private(e, i, u, v, k) num_threads(ntEK)
    for (e = 0; e < E; e++){
        u = Eu[e]*K;
        v = Ev[e]*K;
        i = e*K;
        for (k = 0; k < K; k++){
            Wu[i] *= Aux[u];
            Wv[i] *= Aux[v];
            i++; u++; v++;
        }
    }

    /**  inverse the approximate of the Hessian  **/
    if (al > ZERO){
        #pragma omp parallel for private(v) num_threads(ntVK)
        for (v = 0; v < V*K; v++){ Ga[v] = ONE/Ga[v]; }
    } /* linear case already inverted */

    /**  convergence condition on the metric  **/
    a = ALMOST_TWO*(TWO - rho);
    if (al == ONE){ /* quadratic loss, L = 1 */
        if (a < ONE){
            #pragma omp parallel for private(v) num_threads(ntVK)
            for (v = 0; v < V*K; v++){ if (Ga[v] > a){ Ga[v] = a; } }
        } /* else Ga already less than 1 */
    }else if (al > ZERO){ /* KLa loss, Lk = max_{0<=p_k<=1} d^2KLa/dp_k^2
                           *              = (a/K + (1-a)q_k)(1-a)^2/(a/K)^2 */
        b = ONE/(al_K_al_1*al_K_al_1);
        #pragma omp parallel for private(v, c) num_threads(ntVK)
        for (v = 0; v < V*K; v++){
            c = a/((al_K + al_1*Q[v])*b);
            if (Ga[v] > c){ Ga[v] = c; }
        }
    } /* linear loss: L = infinity */

    /**  precompute some quantities  **/
    if (al > ZERO){ /* weights and thresholds for d1 prox */
        #pragma omp parallel for private(e, i, u, v, k, a, b) num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            a = La_d1[e];
            for (k = 0; k < K; k++){
                W_d1u[i] = Wu[i]/Ga[u];
                W_d1v[i] = Wv[i]/Ga[v];
                b = W_d1u[i] + W_d1v[i];
                Th_d1[i] = a*b/(W_d1u[i]*W_d1v[i]);
                W_d1u[i] /= b;
                W_d1v[i] /= b;
                i++; u++; v++;
            }
        }
    } /* linear loss: weights are all 1/2 and thresholds are all 2 */
    /* metric and first order information */
    if (al == ZERO){ /* linear loss, grad = -Q */
        #pragma omp parallel for private(v) num_threads(ntVK)
        for (v = 0; v < V*K; v++){ GaQ[v] = Ga[v]*Q[v]; }
    }else if (al == ONE){ /* quadratic loss, GaQ is a backup for Ga */
        for (v = 0; v < V*K; v++){ GaQ[v] = Ga[v]; }
    }else{ /* dKLa/dp_k = -(a/K + (1-a)q_k)(1-a)/(a/K + (1-a)p_k) */
        #pragma omp parallel for private(v) num_threads(ntVK)
        for (v = 0; v < V*K; v++){ GaQ[v] = Ga[v]*(al_K + al_1*Q[v]); }
    }

    if (Zu != NULL){ /**  update auxiliary variables  **/
        #pragma omp parallel for private(e, i, u, v, k) num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            for (k = 0; k < K; k++){ 
                if (al == ZERO){ /* linear loss, grad = -Q */
                    Zu[i] = P[u] + GaQ[u] - (Ga[u]/Wu[i])*Zu[i];
                    Zv[i] = P[v] + GaQ[v] - (Ga[v]/Wv[i])*Zv[i];
                }else if (al == ONE){ /* quadratic loss, grad = P - Q */
                    Zu[i] = P[u] - Ga[u]*(P[u] - Q[u] + Zu[i]/Wu[i]);
                    Zv[i] = P[v] - Ga[v]*(P[v] - Q[v] + Zv[i]/Wv[i]);
                }else{ /* dKLa/dp_k = -(a/K + (1-a)q_k)(1-a)/(a/K + (1-a)p_k) */
                    Zu[i] = P[u] + GaQ[u]/(al_K_al_1 + P[u]) - (Ga[u]/Wu[i])*Zu[i];
                    Zv[i] = P[v] + GaQ[v]/(al_K_al_1 + P[v]) - (Ga[v]/Wv[i])*Zv[i];
                }
                i++; u++; v++;
            }
        }
    }

    /** normalize metric to avoid machine precision trouble
     ** when projecting onto simplex  **/
    #pragma omp parallel for private(u, v, k, a) num_threads(ntVK)
    for (u = 0; u < V; u++){
        v = u*K;
        a = Ga[v];
        for (k = 1; k < K; k++){ if (Ga[v+k] > a){ a = Ga[v+k]; } }
        for (k = 0; k < K; k++){ Ga[v+k] /= a; }
    }
}

template <typename real>
void PFDR_graph_loss_d1_simplex(const int K, const int V, const int E, \
    const real al, real *P, const real *Q, \
    const int *Eu, const int *Ev, const real *La_d1, \
    const real rho, const real condMin, \
    real difRcd, const real difTol, const int itMax, int *it, \
    real *Obj, real *Dif, const int verbose)
/* 18 arguments */
{
    /***  initialize general variables  ***/
    if (verbose){ printf("Initializing constants and variables... "); FLUSH; }
    int u, v, e, i, k; /* index edges and vertices */
    real a, b, c; /* general purpose temporary real scalars */
    const real one = ONE; /* argument for simplex projection */
    real al_K, al_1, al_K_al_1;
    if (ZERO < al && al < ONE){ /* constants for KLa loss */
        al_K = al/K;
        al_1 = ONE - al;
        al_K_al_1 = al_K/al_1;
    }

    /***  control the number of threads with Open MP  ***/
    const int ntVK = compute_num_threads(V*K);
    const int ntEK = compute_num_threads(E*K);
    const int ntKE = (ntEK < K) ? ntEK : K;

    /**  allocates general purpose arrays  **/
    real *Ga = (real*) malloc(K*V*sizeof(real)); /* descent metric */
    real *GaQ = (real*) malloc(K*V*sizeof(real)); /* metric and first order information */
    /* auxiliary variables for generalized forward-backward */
    real *Zu = (real*) malloc(K*E*sizeof(real));
    real *Zv = (real*) malloc(K*E*sizeof(real));
    /* splitting weights for generalized forward-backward */
    real *Wu = (real*) malloc(K*E*sizeof(real));
    real *Wv = (real*) malloc(K*E*sizeof(real));
    real *W_d1u, *W_d1v, *Th_d1;
    if (al > ZERO){ /* weights and thresholds for d1 prox */
        W_d1u = (real*) malloc(K*E*sizeof(real));
        W_d1v = (real*) malloc(K*E*sizeof(real));
        Th_d1 = (real*) malloc(K*E*sizeof(real));
    }else{
        W_d1u = W_d1v = Th_d1 = NULL;
    }
    /* initialize p *//* assumed already initialized */
    /* initialize, for all i, z_i = x */
    #pragma omp parallel for private(e, u, v, k, i) num_threads(ntEK)
    for (e = 0; e < E; e++){
        u = Eu[e]*K;
        v = Ev[e]*K;
        i = e*K;
        for (k = 0; k < K; k++){
            Zu[i] = P[u];
            Zv[i] = P[v];
            i++; u++; v++;
        }
    }
    if (verbose){ printf("done.\n"); FLUSH; }

    /***  preconditioning  ***/
    if (verbose){ printf("Preconditioning... "); FLUSH; }
    preconditioning<real>(K, V, E, al, P, Q, Eu, Ev, La_d1, Ga, GaQ, \
                      NULL, NULL, Wu, Wv, W_d1u, W_d1v, Th_d1, rho, condMin);
    if (verbose){ printf("done.\n"); FLUSH; }

    /***  forward-Douglas-Rachford  ***/
    if (verbose){ printf("Preconditioned forward-Douglas-Rachford algorithm\n"); FLUSH; }
    /* initialize */
    int itMsg, it_ = 0;
    real dif, *P_ = NULL; /* store last iterate */
    char msg[256];
    dif = (difTol > difRcd) ? difTol : difRcd;
    if (difTol > ZERO || difRcd > ZERO || Dif != NULL){
        if (difTol >= ONE){
            P_ = (real*) malloc(V*sizeof(real));
            /* compute maximum-likelihood labels */
            #pragma omp parallel for private(u, v, k, a) num_threads(ntVK)
            for (u = 0; u < V; u++){
                v = u*K;
                a = P[v];
                P_[u] = (real) 0;
                for (k = 1; k < K; k++){
                    if (P[v+k] > a){
                        a = P[v+k];
                        P_[u] = (real) k;
                    }
                }
            }
        }else{
            P_ = (real*) malloc(K*V*sizeof(real));
            for (v = 0; v < K*V; v++){ P_[v] = P[v]; }
        }
    }
    if (verbose){
        msg[0] = '\0';
        itMsg = 0;
    }

    /***  main loop  ***/
    while (true){

        /**  objective functional value  **/
        if (Obj != NULL){ 
            a = ZERO;
            if (al == ZERO){ /* linear loss */
                #pragma omp parallel for private(v) reduction(+:a) num_threads(ntVK)
                for (v = 0; v < V*K; v++){ a -= P[v]*Q[v]; }
            }else if (al == ONE){ /* quadratic loss */
                #pragma omp parallel for private(v, c) reduction(+:a) num_threads(ntVK)
                for (v = 0; v < V*K; v++){
                    c = P[v] - Q[v];
                    a += c*c;
                }
                a *= HALF;
            }else{ /* KLa loss */
                #pragma omp parallel for private(u, v, k, c) reduction(+:a) num_threads(ntVK)
                for (u = 0; u < V; u++){
                    v = u*K;
                    /* KLa(p,x) = sum_k (a/K + (1-a)q_k) log((a/K + (1-a)q_k)/(a/K + (1-a)p_k)) */
                    for (k = 1; k < K; k++){
                        c = al_K + al_1*Q[v];
                        a += c*log(c/(al_K + al_1*P[v]));
                        v++;
                    }
                }
            }
            Obj[it_] = a;
            /* ||x||_{d1,La_d1} */
            a = ZERO;
            #pragma omp parallel for private(e, u, v, b, c) reduction(+:a) num_threads(ntEK)
            for (e = 0; e < E; e++){
                u = Eu[e]*K;
                v = Ev[e]*K;
                b = ZERO;
                for (k = 0; k < K; k++){
                    c = P[u] - P[v];
                    if (c < ZERO){ c = -c; }
                    b += c;
                    u++; v++;
                }
                a += La_d1[e]*b;
            }
            Obj[it_] += a;
        }

        /**  progress and stopping criterion  **/
        if (verbose && itMsg++ == verbose){
            print_progress<real>(msg, it_, itMax, dif, difTol, difRcd);
            itMsg = 1;
        }
        if (it_ == itMax || dif < difTol){ break; }

        /**  reconditioning  **/
        if (dif < difRcd){
            if (verbose){
                print_progress<real>(msg, it_, itMax, dif, difTol, difRcd);
                printf("Reconditioning... ");
                FLUSH;
                msg[0] = '\0';
            }
            preconditioning<real>(K, V, E, al, P, Q, Eu, Ev, La_d1, Ga, GaQ, \
                            Zu, Zv, Wu, Wv, W_d1u, W_d1v, Th_d1, rho, condMin);
            difRcd *= TENTH;
            if (verbose){ printf("done.\n"); FLUSH; }
        }

        /**  forward and backward steps on auxiliary variables  **/
        #pragma omp parallel for private(e, u, v, k, i, a, b, c) num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            for (k = 0; k < K; k++){
                /* explicit step */
                if (al == ZERO){ /* linear loss, grad = -Q */
                    a = TWO*P[u] + GaQ[u] - Zu[i];
                    b = TWO*P[v] + GaQ[v] - Zv[i];
                }else if (al == ONE){ /* quadratic loss, grad = P - Q */
                    a = TWO*P[u] - GaQ[u]*(P[u] - Q[u]) - Zu[i];
                    b = TWO*P[v] - GaQ[v]*(P[v] - Q[v]) - Zv[i];
                }else{ /* dKLa/dp_k = -(a/K + (1-a)q_k)(1-a)/(a/K + (1-a)p_k) */
                    a = TWO*P[u] + GaQ[u]/(al_K_al_1 + P[u]) - Zu[i];
                    b = TWO*P[v] + GaQ[v]/(al_K_al_1 + P[v]) - Zv[i];
                }
                /* implicit step */
                if (al == ZERO){ /* weights are all 1/2 and thresholds are all 2 */
                    c = HALF*(a + b); /* weighted average */
                    a = a - b; /* finite difference */
                    /* soft thresholding, update and relaxation */
                    if (a > TWO){
                        a = HALF*(a - TWO);
                        Zu[i] += rho*(c + a - P[u]);
                        Zv[i] += rho*(c - a - P[v]);
                    }else if (a < -TWO){
                        a = HALF*(a + TWO);
                        Zu[i] += rho*(c + a - P[u]);
                        Zv[i] += rho*(c - a - P[v]);
                    }else{
                        Zu[i] += rho*(c - P[u]);
                        Zv[i] += rho*(c - P[v]);
                    }
                }else{
                    c = W_d1u[i]*a + W_d1v[i]*b; /* weighted average */
                    a = a - b; /* finite difference */
                    /* soft thresholding, update and relaxation */
                    if (a > Th_d1[i]){
                        a -= Th_d1[i];
                        Zu[i] += rho*(c + W_d1v[i]*a - P[u]);
                        Zv[i] += rho*(c - W_d1u[i]*a - P[v]);
                    }else if (a < -Th_d1[i]){
                        a += Th_d1[i];
                        Zu[i] += rho*(c + W_d1v[i]*a - P[u]);
                        Zv[i] += rho*(c - W_d1u[i]*a - P[v]);
                    }else{
                        Zu[i] += rho*(c - P[u]);
                        Zv[i] += rho*(c - P[v]);
                    }
                }
                i++; u++; v++;
            }
        }

        /** average **/
        for (v = 0; v < V*K; v++){ P[v] = ZERO; }
        /* this task cannot be easily parallelized along the edges */
        #pragma omp parallel for private(k, e, i) num_threads(ntKE)
        for (k = 0; k < K; k++){
            i = k;
            for (e = 0; e < E; e++){
                P[Eu[e]*K+k] += Wu[i]*Zu[i];
                P[Ev[e]*K+k] += Wv[i]*Zv[i];
                i += K;
            }
        }

        /**  projection on simplex  **/
        proj_simplex_metric<real>(P, Ga, K, V, V, &one, 1);

        /**  iterate evolution  **/
        if (difTol > ZERO || difRcd > ZERO || Dif != NULL){
            dif = ZERO;
            if (difTol >= ONE){
                #pragma omp parallel for private(u, v, k, i, a) reduction(+:dif) num_threads(ntVK)
                for (u = 0; u < V; u++){
                    v = u*K;
                    /* get maximum likelihood label */
                    a = P[v];
                    i = 0;
                    for (k = 1; k < K; k++){
                        if (P[v+k] > a){
                            a = P[v+k];
                            i = k;
                        }
                    }
                    /* compare with previous and update */
                    a = (real) i;
                    if (a != P_[u]){
                        dif += ONE;
                        P_[u] = a;
                    }
                }
            }else{
                /* max reduction available in C since OpenMP 3.1 and gcc 4.7 */
                #pragma omp parallel for private(v, a) reduction(max:dif) num_threads(ntVK)
                for (v = 0; v < V*K; v++){
                    a = P_[v] - P[v];
                    if (a < ZERO){ a = -a; }
                    if (a > dif){ dif = a; }
                    P_[v] = P[v];
                }
            }
            if (Dif != NULL){ Dif[it_] = dif; }
        }

        it_++;
    } /* endwhile (true) */

    /* final information */
    *it = it_;
    if (verbose){
        print_progress<real>(msg, it_, itMax, dif, difTol, difRcd);
        FLUSH;
    }

    /* free stuff */
    free(Ga);
    free(GaQ);
    free(Zu);
    free(Zv);
    free(Wu);
    free(Wv);
    free(W_d1u);
    free(W_d1v);
    free(Th_d1);
    free(P_);
}

/* instantiate for compilation */
template void PFDR_graph_loss_d1_simplex<float>(const int, const int, const int, \
        const float, float*, const float*, const int*, const int*, \
        const float*, const float, const float, float, const float, \
        const int, int*, float*, float*, const int);

template void PFDR_graph_loss_d1_simplex<double>(const int, const int, const int, \
        const double, double*, const double*, const int*, const int*, \
        const double*, const double, const double, double, const double, \
        const int, int*, double*, double*, const int);
