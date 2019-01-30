function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% fprintf('theta size: %f*%f \n', size(all_theta,1), size(all_theta,2));
% fprintf('X size: %f*%f \n', size(X,1), size(X,2));

all_pred = sigmoid(all_theta * X');
% fprintf('all_pred size: %f*%f \n', size(all_pred,1), size(all_pred,2));
[max_p, index_p] = max(all_pred);
p = index_p';
% p = index_p - 1;
% p((p==0)) = 10;
% fprintf('p size: %f*%f \n', size(p,1), size(p,2));
% test = [p,y]


% =========================================================================


end
