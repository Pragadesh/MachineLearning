function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


hyp = X * theta;

modTheta = theta;
modTheta(1) = 0;

cost1 = (sum((hyp - y) .^ 2 ) / (2 * m));
reg = sum(modTheta .^ 2) * lambda / (2 * m);
J = cost1 + reg;

grad1 = X' * (hyp - y);
grad2 = (lambda * modTheta) / m;

grad = (grad1 / m) + grad2;

% =========================================================================


end
