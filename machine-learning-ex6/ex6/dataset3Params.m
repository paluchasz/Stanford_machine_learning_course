function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Values I should try are e.g 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
values_to_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% initialise results which will be a matrix with rows (C_temp, sigma_temp, error)
results = []; 

for C_temp = values_to_try,
  for sigma_temp = values_to_try,
    
    % Train model with temporary C and sigma values, then predict, the find the error
    model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    
    % add C_temp sigma_temp and error to my results matrix 
    results = [results; C_temp sigma_temp error];
    
  end
end


min_err = min(results(:,3)); 
% finds minimum value of 3rd column - the error
row_of_min_err = find(results(:,3) == min_err); 
% returns index of the row which has the min error

C = results(row_of_min_err, 1);
sigma = results(row_of_min_err, 2);




% =========================================================================

end
