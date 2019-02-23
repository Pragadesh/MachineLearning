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

C_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

lowest_model_error = 100;



for C_elem = C_arr
  for sigma_elem = sigma_arr

    model= svmTrain(X, y, C_elem, @(x1, x2) gaussianKernel(x1, x2, sigma_elem));
    predictions = svmPredict(model, Xval);
    model_error = mean(double(predictions ~= yval));
    
    if (model_error < lowest_model_error)
      lowest_model_error = model_error;
      C = C_elem;
      sigma = sigma_elem;
    endif
    
  endfor
endfor

fprintf("C: %f\n", C);
fprintf("sigma: %f\n", sigma);
fprintf("model_error: %f\n", lowest_model_error);


% =========================================================================

end
