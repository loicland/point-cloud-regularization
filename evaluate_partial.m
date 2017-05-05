function partial_score = evaluate_partial(GT_labels, est_proba...
    , first_bin, num_bin, file_name)
%compute the partial accuracy/coverage curves
%Input
%GT_labels = the ground truth
%est_proba = the estimated probability
%first_bin = the minimum coverage (default = 70)
%num_bin   = the number of points in the curve (default = 50)
%file_name = if present, will write the result in a csv file
if (nargin < 3)
  first_bin = .70;  
end
if (nargin < 4)
  num_bin = 50;  
end
%---compute and order the entropy------------------------------------------
entropy = -est_proba.*log(est_proba);
entropy(est_proba==0) = 0;
entropy = sum(entropy,2)/log(size(est_proba,2));
[dump, order] = sort(entropy);
%---compute and order the entropy------------------------------------------
partial_score = nan(num_bin, 1);
ratioArray    = linspace(first_bin, 1, num_bin);
%---compute the partial accuracy curve-------------------------------------
[dump, est_labels] = max(est_proba,[],2);
est_labels         = uint8(est_labels);
for i = num_bin:-1:1
    ratio = ratioArray(i);
    confidence = order(1:ceil(numel(order) * ratio));
    [dump, acc] = evaluate(GT_labels(confidence), est_labels(confidence)...
        , false);
    partial_score(i) = acc;
end
%---write a csv file-------------------------------------------------------
if (nargin == 5)
    fid = fopen([sprintf('./results/%s',file_name), '.csv'], 'w');
    fprintf( fid, 'P        F   \n');
    for i=1:num_bin
         fprintf( fid, '%3.2f    %3.2f\n', 100*ratioArray(i)...
             , 100*partial_score(i));
    end
    fclose( fid );
end