function p_regularized = PFDR(initial_p, graph, lambda, fidelity)
%Preconditonned Forward Douglas Radchford Algorithm to solve
%TV penalized simplex bound energies
%INPUT
%inital_labeling = classification to regularize
%graph  = the adjacency structure
%fidelity = which fidelity fucntion to use (default = 1)
%	0 : linear
%	1 : quadratic  
%	2 : KL with 0.05 uniform smoothing
%	3 : loglinear with 0.05 uniform smoothing
%lambda      : regularization strength (default = 1)
%benchMarking: if true will return the energy and time for the algorithm
%            : stopped after 1 ... maxIte iteration, starting from zero 
%              each time
%OUTPUT
%p_regularized = the regularized probability
%loic landrieu 2016
%
%When using this method you must cite:
%
%A Note on the Forward-Douglas--Rachford Splitting for Monotone Inclusion
%and Convex Optimization.
%Raguet, H. (2017).
smoothing = 0.05;
if (nargin < 3)
    lambda = 1;
end
if (nargin < 4)
    fidelity = 1;
end
nClasses  = size(initial_p,2);
switch fidelity
    case 0
       p_regularized = PFDR_graph_loss_d1_simplex_mex(initial_p', 0 ,...
        int32(graph.source) ,  int32(graph.target)...
        , graph.edge_weight * lambda, 1, 0.2, 1e-1, 1, 100, 0);
     case 1
       p_regularized = PFDR_graph_loss_d1_simplex_mex(initial_p', 1 ,...
        int32(graph.source) ,  int32(graph.target)...
        , graph.edge_weight * lambda, 1, 0.2, 1e-1, 1, 100, 0);
    case 2
      p_regularized = PFDR_graph_loss_d1_simplex_mex(initial_p', smoothing ,...
        int32(graph.source) ,  int32(graph.target)...
        , graph.edge_weight * lambda, 1, 0.2, 1e-1, 1, 100, 0);
    case 3
       p_regularized = PFDR_graph_loss_d1_simplex_mex(log(initial_p' ...
            * (1-smoothing + smoothing/nClasses)), 0, int32(graph.source)...
            ,  int32(graph.target), graph.edge_weight * lambda ...
            , 1, 0.2, 1e-1, 1, 200, 0);
end
    
