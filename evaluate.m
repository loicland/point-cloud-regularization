function [mean_F1_score, accuracy ]  = evaluate(GT_labels, class_est, verbose)
if (nargin == 2)
    verbose  = true;
end
% evaluate classifier
CM = confusionmat(GT_labels,class_est);
accuracy = trace(CM)/sum(sum(CM));      % observed accuracy
% more detailed evaluation
num_classes = size(CM,1);
TP = zeros(num_classes,1);
FP = zeros(num_classes,1);
FN = zeros(num_classes,1);
TN = zeros(num_classes,1);
recall = zeros(num_classes,1);
precision = zeros(num_classes,1);
NPV = zeros(num_classes,1);
FDR = zeros(num_classes,1);
F1_score = zeros(num_classes,1);
observed_accuracy = accuracy;
proportion_of_examples_belonging_to_class_i = sum(CM,2);                % summing entries in a row 
proportion_of_examples_assigned_by_classifier_to_class_i = sum(CM,1);   % summing entries in a column 
sum_rands = 0;
for i=1:num_classes
    TP(i,1) = CM(i,i);                              % true positives (TP)
    FP(i,1) = sum(CM(:,i)) - CM(i,i);               % false positives (FP)
    FN(i,1) = sum(CM(i,:)) - CM(i,i);               % false negatives (FN)
    TN(i,1) = sum(sum(CM)) - TP(i,1) - FP(i,1) - FN(i,1);   % true negatives (TN) 
    recall(i,1) = TP(i,1) / (TP(i,1) + FN(i,1));    % recall = true positive rate (TPR) 
    precision(i,1) = TP(i,1) / (TP(i,1) + FP(i,1)); % precision = positive predictive value (PPV)
    NPV(i,1) = TN(i,1) / (TN(i,1) + FN(i,1));       % negative predictive value (NPV)
    FDR(i,1) = FP(i,1) / (FP(i,1) + TP(i,1));       % false discovery rate (FDR)
    F1_score(i,1) = 2 * precision(i,1) * recall(i,1) / (precision(i,1) + recall(i,1));  % F-score 
    sum_rands = sum_rands + proportion_of_examples_belonging_to_class_i(i)*proportion_of_examples_assigned_by_classifier_to_class_i(i);
end
random_classifier_accuracy = sum_rands / ( sum(sum(CM))^2 );
kappa = (observed_accuracy - random_classifier_accuracy) / (1 - random_classifier_accuracy);
mean_recall = sum(recall) / num_classes;
mean_precision = sum(precision) / num_classes;
mean_F1_score = nanmean(F1_score);

if (verbose)
fprintf('CLASSE   PRECISION   RECALL   F_SCORE\n');
for i=1:num_classes
fprintf('%4.0f        %4.1f      %4.1f     %4.1f\n', i, precision(i)*100,  recall(i)*100, F1_score(i)*100);    
end
fprintf('MEAN        %4.1f      %4.1f     %4.1f\n', mean_precision*100,  mean_recall*100, mean_F1_score*100);   
fprintf('KAPPA = %2.1f   ACCURACY = %2.1f \n\n', kappa*100, accuracy * 100);

fprintf('%4.1f & %4.1f',accuracy * 100, mean_F1_score*100);
if (0)
    for i=1:num_classes
        fprintf(' & %4.1f & %4.1f &%4.1f', precision(i)*100,  recall(i)*100, F1_score(i)*100); 
    end
end
if (1)
    for i=1:num_classes
        fprintf(' & %4.1f', F1_score(i)*100); 
    end
end
fprintf('\n');
end
