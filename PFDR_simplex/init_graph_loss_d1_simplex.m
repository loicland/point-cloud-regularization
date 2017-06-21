clear all;

%%%  parameters  %%%
workDir = '~/Recherche/Optimization/Graph_loss_d1_simplex/';
debugmode = false
saveRes = false
% dataset = 'validationSet'
% dataset = 'trainingSet'
% dataset = 'bigSet'
dataset = 'bigSet2'
% dataset = 'oakland_bug'
%% trainingSet
% 1 - wire (2571)
% 2 - poles (1086)
% 3 - facades (4713)
% 4 - ground (14121)
% 5 - vegetation (14441)
classNames = strvcat('wires', 'poles', 'facades', 'ground', 'vegetation');
classColors = [000 000 255;
               255 000 000;
               200 200 255;
               255 190 000; 
               000 255 065];

%%%  initialize and load data  %%%
cd(workDir);
load(dataset, 'c', 'Eu', 'Ev', 'La');
K = size(c, 2);
La = repmat(La, [1 K]);

%%%  optimization parameters  %%%
rho = 1; % relaxation parameter for fdr, 0 < rho < 2
w_simplex = .5; % weights for simplex constraint in GFB, 0 < w_simplex < 1
itMax = 100; % maximum number of iterations
difTol = 1e-2; % minimum iterate evolution
difRcd = 0; % reconditioning criterion
condMin = 1; % ensure stability of the preconditioning
verbose = 10 % number of iterations between two progress messages
