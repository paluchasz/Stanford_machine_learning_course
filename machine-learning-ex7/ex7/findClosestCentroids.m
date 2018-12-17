function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);

for i = 1:m,
  
  % Find coordinates of data point:
  x = X(i, :);
  
  % Initialize an array which will store distances to each cluster
  distances = [];
  
  for j = 1:K,
    % find coordinates of cluster
    c = centroids(j, :);
    
    distance = (x-c) * (x - c)';
    % I transpose the second one because x and c are row vectors

    distances = [distances; distance];
    % Append to distances for each centroid
  end
  
  min_distance = min(distances); % find minimum distance
  cluster_id = find(distances == min_distance); % and which centroid that corresponds to
  idx(i) = cluster_id(1); % in case there is more than one optimum distance just pick the first
  
end

   
    
    




% =============================================================

end

