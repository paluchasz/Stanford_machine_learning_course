function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% check my notebook for workings out!

% Add ones to the X data matrix, this takes care of the bias in a_1/x
X = [ones(m, 1) X];

% now need to go through all training examples
for i = 1:m,
  % selecting a training example
  x = X(i,:)';
  z_2 = Theta1 * x;
  a_2 = sigmoid(z_2);
  %Need to add bias to a_2:
  a_2 = [1; a_2];
  z_3 = Theta2 * a_2;
  h = sigmoid(z_3);
  % h is now a 10x1 vector of probabilities, we need to take the maximum of 
  % the probabilities for each training example and the index of the highest
  % probability to p since that will correspond to the digit prediciton.
  maximum = max(h); % finds the max
  prediction = find(h == maximum); % finds the index
  p(i) = prediction;
end






% =========================================================================


end
