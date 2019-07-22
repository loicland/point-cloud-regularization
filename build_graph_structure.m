function graph = build_graph_structure(filename, k, dist_cap ...
    , edge_weight_mode)
%compute the knn structure
% d0  = average distance between points in the full graph
%INPUT
%string filename
%int k = number of nodes (default is 10)
%dist_cap = pruning threshold = dist_cap * d0.  A value of 0 (default)
%means no pruning
%single edge_weight_mode = weighting mode of the edges
%   c = 0 : constant weight (default)
%   c > 0 : linear weight w = 1/(d/d0 + c) 
%   c < 0 : exponential weightw = exp(-d/(c * d0))
%OUTPUT
%struct graph =  a structure with the following fields:
%single matrix XYZ : coordinates of each point
%int32 vectors source, target: index of the vertices constituting the
%edges
%single vector edge_weight: the edge of the  
if (nargin < 2)
    k = 10;
end
if (nargin < 3)
    dist_cap = 0;
end
if (nargin < 4)
    edge_weight_mode = 0;
end
graph= struct;
ply_data = pcread(filename);
graph.XYZ = ply_data.Location;
clear('ply_data');
n_point = size(graph.XYZ,1);
%---compute full adjacency graph-------------------------------------------
[neighbors,distance] = knnsearch(graph.XYZ,graph.XYZ,'K', k+1);
neighbors = neighbors(:,2:end);
distance  = distance(:,2:end);
d0     = mean(distance(:));
source      = reshape(repmat(1:n_point, [k 1]), [1 (k * n_point)])';
target      = reshape(neighbors', [1 (k * n_point)])';
%---edge_weight computation------------------------------------------------
edge_weight = ones(size(distance));
if (edge_weight_mode>0)
    edge_weight = 1./(distance / d0 + edge_weight_mode);
elseif (edge_weight_mode<0)
    edge_weight = exp(distance / (d0 * edge_weight_mode));
end
edge_weight = reshape(edge_weight', [1 (k * n_point)])';
%---pruning----------------------------------------------------------------
pruned = false(size(distance));
if (dist_cap>0)
    pruned = distance > dist_cap * d0;
end
pruned = reshape(pruned, [1 (k * n_point)])';
%---remove self edges and pruned edges-------------------------------------
selfedge = source==target;
to_remove = selfedge + pruned;
source      = source(~to_remove)-1;
target      = target(~to_remove)-1;
edge_weight = edge_weight(~to_remove);
%---symetrizing the graph -------------------------------------------------
double_edges_coord  = [[source;target],[target;source]];
double_edges_weight = [edge_weight;edge_weight];                 
[edges_coord, order] = unique(double_edges_coord, 'rows');
edges_weight = double_edges_weight(order);
%---filling the structure -------------------------------------------------
graph.source      = int32(edges_coord(:,1));
graph.target      = int32(edges_coord(:,2));
graph.edge_weight = single(edges_weight);

