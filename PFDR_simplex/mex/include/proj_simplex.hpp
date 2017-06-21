/*==================================================================
 * Orthogonal projection over the simplex:
 *      for all n, x_n >= 0; and sum_n x_n = a.
 *
 * Work in-place, independently for each column.
 *
 * Parallel implementation with OpenMP API.
 * 
 * Hugo Raguet 2016
 *================================================================*/
#ifndef PROJ_SIMPLEX_H
#define PROJ_SIMPLEX_H

template <typename real>
void proj_simplex(real *X, const int D, const int N, const real *A, const int na);
/* 5 arguments
 * X  - array of N D-dimensionnal vectors, D-by-N array, column major format
 * D  - dimensionality of each vectors
 * N  - number of input vectors
 * A  - for each vector, the desired sum of the coordinates of the vector, array
 *      of length na
 * na - the number of specified desired sums. if less than N, all remaining are
 *      equal to the last specified sum */

/*==================================================================
 * Orthogonal projection over the simplex:
 *      for all n, x_n >= 0; and sum_n x_n = a,
 * within a diagonal metric defined by 1./m as,
 *      <x,y>_{1/m} = <x, diag(1./m) y> = sum_n x_n y_n / m_n.
 * i.e. m is the vector of the /inverses/ of the diagonal entries of the 
 * matrix of the desired metric.
 *================================================================*/
template <typename real>
void proj_simplex_metric(real *X, const real *M, const int D, const int N, \
                                     const int nm, const real *A, const int na);
/* +2 = 7 arguments
 * M  - for each vector, (inverse terms of) a diagonal metric, D-by-nm array,
 *      column major formats
 * nm - number of specified metrics.if less than N, all remaining are
 *      equal to the last specified metric */

#endif
