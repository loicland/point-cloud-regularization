%        [P, it, Obj, Dif] = PFDR_graph_loss_d1_simplex_mex(Q, al, Eu, Ev, La_d1, rho, condMin, difRcd, difTol, itMax, verbose)
%
% Minimize functionals on graphs of the form:
%
%        F(p) = f(p) + ||p||_{d1,La_d1} + i_{simplex}(p)
%
% where for each vertex, p_v is a vector of length K,
%       f is a data-fidelity loss (depending on q and parameter al, see below)
%       ||p||_{d1,La_d1} = sum_{k, uv in E} la_d1_uvk |p_uk - p_vk|,
%   and i_{simplex} is the standard simplex constraint over each vertex,
%       for all v, (for all k, p_vk >= 0) and sum_k p_vk = 1,
%
% using preconditioned forward-Douglas-Rachford algorithm.
%
% INPUTS: (warning: real numeric type is either single or double, not both)
% Q        - observed probabilities, K-by-V array (real)
% al       - scalar defining the data-fidelity loss function
%            al = 0, linear:
%                      f(p) = - <q, p>,
%              with  <q, p> = sum_{k,v} q_kv p_kv,
%            al = 1, quadratic:
%                      f(p) = 1/2 ||q - p||_{l2}^2,
%              with  ||q - p||_{l2}^2 = sum_{k,v} (q_kv - p_kv)^2.
%            0 < al < 1, smoothed Kullback-Leibler divergence:
%                      f(p) = sum_v KLa(q_v||p_v),
%              with KLa(q_v||p_v) = KL(au + (1-a)q_v || au + (1-a)p_v),
%              where KL is the regular Kullback-Leibler divergence,
%                    u is the uniform discrete distribution over {1,...,K},
%                    and a = al is the smoothing parameter.
%              Up to a constant - H(au + (1-a)q_v))
%                  = sum_{k} (a/K + (1-a)q_v) log(a/K + (1-a)q_v),
%              we have KLa(q_v||p_v)
%                  = - sum_{k} (a/K + (1-a)) q_v log(a/K + (1-a)p_v).
% Eu       - for each edge, index of one vertex, array of length E (int32)
% Ev       - for each edge, index of the other vertex, array of length E (int32)
%            Every vertex should belong to at least one edge. If it is not the
%            case, a workaround is to add an edge from the vertex to itself
%            with a nonzero penalization coefficient.
% La_d1    - d1 penalization coefficients for each edge, array of length E (real)
% rho      - relaxation parameter, 0 < rho < 2
%            1 is a conservative value; 1.5 often speeds up convergence
% condMin  - positive parameter ensuring stability of preconditioning
%            1 is a conservative value; 0.1 or 1e-2 might enhance preconditioning
% difRcd   - reconditioning criterion on iterate evolution.
%            If difTol < 1, reconditioning occurs if all coordinates of P 
%            change by less than difRcd. difRcd is then divided by 10.
%            If difTol >= 1, reconditioning occurs if less than
%            difRcd maximum-likelihood labels have changed. difRcd is then 
%            divided by 10.
%            0 is a conservative value, 10*difTol or 1e2*difTol might
%            speeds up convergence. reconditioning might temporarily draw 
%            minimizer away from solution; it is advised to monitor objective
%            value when using reconditioning
% difTol   - stopping criterion on iterate evolution.
%            If  difTol < 1, algorithm stops if all coordinates of P change by 
%            less than difTol. If difTol >= 1, algorithm stops if less than
%            difTol maximum-likelihood labels have changed.
% itMax    - maximum number of iterations
% verbose  - if nonzero, display information on the progress, every 'verbose'
%            iterations
%
% OUTPUTS:
% P   - minimizer, K-by-V array (real)
% it  - actual number of iterations performed
% Obj - if requested, the values of the objective functional along
%       iterations (it+1 values)
% Dif - if requested, the iterate evolution along iterations (see difTol)
%
% Parallel implementation with OpenMP API.
%
% Reference: H. Raguet, A note on the forward-Douglas-Rachford splitting
% algorithm, and application to convex optimization, to appear.
%
% Hugo Raguet 2016
