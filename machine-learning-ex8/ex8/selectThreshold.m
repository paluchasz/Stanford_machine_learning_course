function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % This vector will predict 1 if the probability is < epsilon (meaning an
    % (anomalous example) and 0 if probability > epsilon (meaning a normal example)
    predictions = pval < epsilon;
    
    % Look at 2x2 table in Week 6, handling skewed data for a reminder of the following:
    
    % True positives (number of occurances when predicted class = 1 and actual class = 1):
    tp = sum((predictions == 1) & (yval == 1));
    % False positives (occurances where predicted class = 1 and actual class = 0):
    fp = sum((predictions == 1) & (yval == 0));
    % False negatives (occurances where predicted = 0 and actual = 1):
    fn = sum((predictions == 0) & (yval == 1));
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    
    F1 = 2 * precision * recall / (precision + recall);










    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
