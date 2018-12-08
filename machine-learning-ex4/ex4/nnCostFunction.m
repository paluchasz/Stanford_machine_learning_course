function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% Add ones to the X data matrix, this takes care of the bias in a_1/x
X = [ones(m, 1) X];

% Intialise J
J = 0;

% now need to go through all training examples and for each h(theta) work 
% out the cost and add to J
for i = 1:m,
  % selecting a training example
  x = X(i,:)';
  a_1 = x;
  
  %%% Feedforward part: %%%
  z_2 = Theta1 * x;
  a_2 = sigmoid(z_2);
  
  %Need to add bias to a_2:
  a_2 = [1; a_2];
  z_3 = Theta2 * a_2;
  h = sigmoid(z_3);
  % h is now a 10x1 vector of probabilities
  
  % Also at the moment y is a mx1 vector, for each of its elements need to convert
  % it to a 10D basis vector, e.g if y(i) = 1 then need (1,0,....0) and if y(i) = 2
  % then need (0,1,0,....0)
  y_output_layer = zeros(num_labels, 1);
  y_output_layer(y(i)) = 1; 

  J_for_one_example = 1/m * (-y_output_layer' * log(h) - (1 -y_output_layer)' * log(1 -h));
  
  J = J + J_for_one_example;
  
  %%% Backpropagation part %%%
  delta_3 = h - y_output_layer;
  z_2 = [1; z_2]; % adding bias to z_2 so dimensions match below:
  delta_2 = Theta2' * delta_3 .* sigmoidGradient(z_2);
  
  delta_2 = delta_2(2:end); %removing the bias error deta_0^(2)
  
  % Keep accumulating the gradients 
  Theta1_grad = Theta1_grad + delta_2 * a_1';
  Theta2_grad = Theta2_grad + delta_3 * a_2';
end

% The final step of computing the gradients
Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

% Regularization terms for J. 
regularization_term = 0;

%Theta1: let i represent rows and j columns; note how we are starting j at 2 to 
% ignore the first column of Theta which corresponds to the bias.
for i = 1:size(Theta1, 1),
  for j = 2:size(Theta1, 2),
    regularization_term = regularization_term + lambda/(2*m) * Theta1(i,j)**2;
  end
end

%Theta2:
for i = 1:size(Theta2, 1),
  for j = 2:size(Theta2, 2),
    regularization_term = regularization_term + lambda/(2*m) * Theta2(i,j)**2;
  end
end

% Regularized J:
J = J + regularization_term;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
