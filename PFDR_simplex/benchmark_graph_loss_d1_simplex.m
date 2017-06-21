%%%  objective  %%%
La_ = La;
al = .2; % parameter for KL smoothing

%%%  optimization parameters  %%%
%%  PFDR  %%
rho = 1.5; % relaxation parameter for fdr, 0 < rho < 2
w_simplex = .5; % weights for simplex constraint in GFB, 0 < w_simplex < 1
itMax = 100; % maximum number of iterations
difTol = 1; % minimum iterate evolution
% difRcd = 1e-3*size(c, 1); % reconditioning criterion
difRcd = 0; % reconditioning criterion
condMin = 1; % ensure stability of the preconditioning
verbose = 1; % number of iterations between two progress messages

%%  CP  %%
CP_difTol = 1e-2;
CP_itMax = 10;
PFDR_rho = 1.5;
PFDR_condMin = 1e-3;
PFDR_difRcd = 0;
PFDR_difTol = 1e-5;
PFDR_itMax = 1e4;
PFDR_verbose = 1e3;

xm = cast(bsxfun(@eq, c, max(c, [], 2)), class(c));

% %{
tic;
[x, it, obj] = PFDR_graph_loss_d1_simplex_mex(xm', 0, Eu, Ev, La_(:,1), rho, condMin, difRcd, difTol, itMax, verbose);
t_2 = toc
it_2 = double(it);
obj_2 = obj(2:it_2+1);
% dif_2 = dif(1:it_2);
tim_2 = (1:it_2)*t_2/it_2;
% tim_2 = 1:it_2;
x_2 = x;
%}

%{
graph = struct;
graph.observation = double(c);
graph.source = double(Eu);
graph.target = double(Ev);
graph.edge_weight = double(La_(:,1));
[l_bk, obj_bk, t_bk] = Boykov(graph, 1, 5);
%}

% leg = {'cond 1e-1', 'cond 1', 'rho 1.5', 'BK'};
% leg = {'cond 1e-1', 'cond 1', 'rho 1.5'};
leg = {'old', 'new'};
cols = jet(length(leg));
clf, i = 1;
semilogy(tim_1, obj_1,    'color', cols(i,:), 'LineWidth', 2); i = i+1;
hold on
semilogy(tim_2, obj_2,  'color', cols(i,:), 'LineWidth', 2); i = i+1;
legend(leg);
ylabel('obj');
axis tight
