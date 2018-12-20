function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


% sum() sums each column of X, then if we transpose we get a vector equivalent to
% summing all vectors x^(i).
mu = 1/m * sum(X)';  

% Once we transpose X each column is an x^(i). Hence the ith column of X' .- mu 
% corresponds to x^i - mu, we then square each term of the matrix in turn and finally
% sum over each row 
temp = (X' .- mu) .^2;
sigma2 = 1/m * sum(temp, 2); % the 2 indicates we are summing over rows not columns







% =============================================================


end
