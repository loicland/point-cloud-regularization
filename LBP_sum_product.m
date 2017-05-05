function [assignment, T] = LBP_sum_product(initial_labeling, graph, lambda, maxIte)
%implement the loopy belief propagation sum product algorithm
%the transition matrix is defined as follow:
%1 on the diagonal and exp(lambda) outside
%INPUT 
%inital_labeling = classification to regularize
%graph  = the adjacency structure
%lambda = the regularization strength (>0)
%maxIte = maximum number of iteration (default = 30)
%
%When using this method you must cite:
%M. Schmidt. UGM: A Matlab toolbox for probabilistic undirected graphical
%models. http://www.cs.ubc.ca/~schmidtm/Software/UGM.html, 2007. 
if (nargin < 4)
    maxIte = 30;
end
nClasses    = size(initial_labeling,2);
%it is necessary to have a symetric structure for this algorithm
adjacency    = sparse(double(graph.source)+1,double(graph.target)+1 ...
             , double(graph.edge_weight));
edgeStruct  = UGM_makeEdgeStruct(adjacency,nClasses,1);
unary       = initial_labeling;
transition      = single(ones(nClasses));
transition(logical(eye(nClasses))) = single(exp(lambda));
binary = repmat(transition, [1 1 edgeStruct.nEdges]);
edgeStruct.useMex  = 1;
edgeStruct.maxIter = int32(maxIte);
tic;
clear('V','adj')
[nodeBelLBP] = UGM_Infer_LBP(unary,binary,edgeStruct);
assignment = nodeBelLBP;
T = toc;