%==========================================================================
%=========== BENCHMARKING THE REGULARIZATION OF SEMANTIC LABELINGS ========
%========================== OF POINT CLOUDS================================
%==========================================================================
%=====================     LOIC LANDRIEU  2017   ==========================
%==========================================================================
%Implementing the methods of the following article
%A structured regularization framework for spatially smoothing semantic
%labelings of 3D point clouds. Landrieu, L., Raguet, H., Vallet, B.,
%Mallet, C., & Weinmann, M. (2017).
%--- dependecies ----------------------------------------------------------
clear all;
addpath('./data')
addpath('./UGM/sub/')
addpath('./UGM/compiled/')
addpath('./UGM/infer/')
addpath('./UGM/decode/')
addpath('./GCMex')
addpath('./PFDR_simplex/mex/bin/')
addpath('./CutPursuit/bin/')
%----build the adjacency graph---------------------------------------------
graph = build_graph_structure('oakland.ply',10,0,0);
%---retrieve labeling with your favorite classifier------------------------
load('oakland_RF', 'initial_classif');
%---if available - load the ground truth here------------------------------
load('oakland_GT', 'ground_truth'); %MUST BE UINT8
%--------------------------------------------------------------------------
%--------------- BENCHMARKING ---------------------------------------------
%--------------------------------------------------------------------------
%--- baseline approach ----------------------------------------------------
[dump, l_baseline] = max(initial_classif,[],2);
evaluate(ground_truth, uint8(l_baseline));
partial_baseline = evaluate_partial(ground_truth, initial_classif,0.7,50);
%---LBP sum product--------------------------------------------------------
[p_lpb_sp, T] = LBP_sum_product(initial_classif, graph, .5, 10);
[dump, l_lpb_sp] = max(p_lpb_sp,[],2);
evaluate(ground_truth, uint8(l_lpb_sp));
partial_lpb_sp = evaluate_partial(ground_truth,p_lpb_sp ,0.7,50);
%---LBP max product--------------------------------------------------------
[l_lpb_mp, T] = LBP_max_product(initial_classif, graph, .5, 20);
[dump, acc_lbp_max] = evaluate(ground_truth, uint8(l_lpb_mp));
%---Alpha-expansion--------------------------------------------------------
[l_lin_potts, T] = alpha_expansion(initial_classif, graph, 0, 1, 5);
[dump, acc_in_potts] = evaluate(ground_truth, uint8(l_lin_potts));
%---Total Variation--------------------------------------------------------
p_kl_tv = PFDR(initial_classif, graph, 0.5, 2);
[dump, l_kl_tv] = max(p_kl_tv,[],1);
evaluate(ground_truth, uint8(l_kl_tv));
partial_kl_tv = evaluate_partial(ground_truth,p_kl_tv' ,0.7,50); 
%---Cut Pursuit------------------------------------------------------------
p_kl_bo = L0_cut_pursuit(initial_classif, graph, 5, 2);
[dump, l_kl_bo] = max(p_kl_bo,[],1);
evaluate(ground_truth, uint8(l_kl_bo));
partial_kl_bo = evaluate_partial(ground_truth,p_kl_bo' ,0.7,50); 
%--------------------------------------------------------------------------
%--------------- VIZUALIZATION --------------------------------------------
%--------------------------------------------------------------------------
clf;hold on;
plot(linspace(70,100,50),partial_baseline,'k-');
plot(linspace(70,100,50),partial_lpb_sp,'k--');
plot([70,100],[acc_lbp_max, acc_lbp_max],'k.');
plot([70,100],[acc_in_potts, acc_in_potts],'r-.');
plot(linspace(70,100,50),partial_kl_tv,'g--');
plot(linspace(70,100,50),partial_kl_bo,'b--');





