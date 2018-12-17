function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i = 1:K,
  
  sel = find(idx == i); 
  % selection of all indices for each centroid, if I had [idx == i] this would give 
  % a binary vector with 1 if that particular training example was added to this centroid
  % so find, finds the indices of these 1s which tells me which examples to average
  
  centroids(i, :) = mean(X(sel, :));
  % X is a matrix so compute the mean for each column and return them in a row vector.
  % Finding the mean of some vectors is the same as finding the mean of the individual 
  % components for all components!

end





% =============================================================


end

