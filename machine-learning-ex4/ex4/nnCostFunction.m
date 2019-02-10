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

A1 = [ones(1,size(X,1));X'];
A2 = Theta1 * A1;
Z2 = sigmoid(A2);
Z2_withUnit = [ones(1,size(Z2,2));Z2];
A3 = Theta2 * Z2_withUnit;
hyp = sigmoid(A3)';

% This does not work since we need inly the sum of the diagonal of the matrix
%Yk = [];
%for classifier = 1:num_labels
%  Yk = [Yk; (y == classifier)'];
%endfor
%cost = ((-Yk*log(hyp)) - ((1-Yk)*log(1-hypK)));


cost = 0;
for classifier = 1:num_labels
  Yk = (y == classifier)';
  hypK = hyp(:,classifier);
  cost = cost + ((-Yk*log(hypK)) - ((1-Yk)*log(1-hypK)));
endfor

J = sum(cost)/m;

Theta1_sq = Theta1;
Theta1_sq(:,1) = zeros(size(Theta1,1), 1);
Theta1_sq = Theta1_sq .^ 2;

Theta2_sq = Theta2;
Theta2_sq(:,1) = zeros(size(Theta2,1), 1);
Theta2_sq = Theta2_sq .^ 2;

reg = sum(sum(Theta1_sq)) + sum(sum(Theta2_sq));

J = J + ((lambda * reg) / (2 * m));

delta1 = zeros(hidden_layer_size, input_layer_size+1);
delta2 = zeros(num_labels, hidden_layer_size+1);

%fprintf("num_labels: %d\n", num_labels);

for t = 1 : m
  A1T = [1;X(t,:)'];
  Z2T = Theta1 * A1T;
  A2T = sigmoid(Z2T);
  A2T_withUnit = [1;A2T];
  Z3T = Theta2 * A2T_withUnit;
  hypT = sigmoid(Z3T);
  yT = zeros(num_labels,1);
  % yT(mod(y(t),num_labels)+1) = 1;
  yT(y(t)) = 1;
  sig3 = hypT - yT;
  sig2 = (Theta2' * sig3) .* sigmoidGradient([1;Z2T]);
  sig2 = sig2(2:end);
  delta2 = delta2 + (sig3 * A2T_withUnit');
  delta1 = delta1 + (sig2 * [1,X(t,:)]);
endfor




Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;
Theta1_grad = (delta1/m) + ((lambda * Theta1_reg)/m);
Theta2_grad = (delta2/m) + ((lambda * Theta2_reg)/m);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
