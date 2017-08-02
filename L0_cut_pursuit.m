function p_regularized = L0_cut_pursuit(initial_p, graph, lambda, fidelity)
%L0-Cut Pursuit to solkve penalization by the boundary length 
%INPUT
%inital_labeling = classification to regularize
%graph  = the adjacency structure
%fidelity = which fidelity fucntion to use (default = 0)
%	0 : linear
%	1 : quadratic  
%	2 : KL with 0.05 uniform smoothing
%	3 : loglinear with 0.05 uniform smoothing
%lambda      : regularization strength (default = 1)
%maxIte      : maximum number of alpha expansion cycle (default = 5)
%benchMarking: if true will return the energy and time for the algorithm
%            : stopped after 1 ... maxIte iteration, starting from zero 
%              each time
%OUTPUT
%p_regularized = the regularized probability
%loic landrieu 2016
%
%When using this method you must cite:
%
%Cut Pursuit: fast algorithms to learn piecewise constant functions on
%general weighted graphs,
%Landrieu, Loic and Obozinski, Guillaume,2016.
smoothing = single(0.05);
if (nargin < 3)
    lambda = 1;
end
if (nargin < 4)
    fidelity = 1;
end
nClasses  = size(initial_p,2);
node_weight = single(ones(size(initial_p,1),1));
switch fidelity
    case 0
       p_regularized = L0_cut_pursuit_mex(initial_p', graph.source...
           ,graph.target, lambda, graph.edge_weight, node_weight...
           , 0, 2, 0);
     case 1
         p_regularized = L0_cut_pursuit_mex(initial_p', graph.source...
           ,graph.target, lambda, graph.edge_weight, node_weight...
           , 1, 2, 0);
    case 2
         p_regularized = L0_cut_pursuit_mex(initial_p', graph.source...
           ,graph.target, lambda, graph.edge_weight, node_weight...
           , smoothing, 2, 0);
    case 3
        p_regularized = L0_cut_pursuit_mex(log(initial_p' ...
            * (1-smoothing + smoothing/nClasses)), graph.source...
           ,graph.target, lambda, graph.edge_weight, node_weight...
           , 0, 2, 0);
end
    
