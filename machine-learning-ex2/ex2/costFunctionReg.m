function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X*theta); % The hypothesis

% Recall regularization term is lambda * sum of theta_j^2 but from i = 1 not 0!
% since theta^T * theta = this sum + theta_0 ^2 we get (recall that we start
% indexing at 1 in octave!):
regularization_term = lambda/(2*m) * (theta' * theta - theta(1)*theta(1)); 

J = 1/m * (-y' * log(h) - (1-y)' * log(1-h)) + regularization_term;
% same as in costFunction.m plus the extra term

unregularized_grad = 1/m * X' * (h - y); %unregularized vector for all gradients

% Regularization term for gradient is lambda/m * theta_j for all j except j=0
% So we want to add a vector of values lambda/m * theta_j except for first element:
regularization_gradient_vector = lambda/m * theta;
regularization_gradient_vector(1) = 0;

grad = unregularized_grad + regularization_gradient_vector;


% =============================================================

end
