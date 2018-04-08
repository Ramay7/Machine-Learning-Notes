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

%fprintf('size(X) = (%d, %d)\n', size(X));
%fprintf('size(y) = (%d, %d)\n', size(y));
%fprintf('size(Xval) = (%d, %d)\n', size(Xval));
%fprintf('size(Yval) = (%d, %d)\n', size(yval));

step = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
J = zeros(size(step, 1));
%fprintf('size(step) = (%d, %d)\n', size(step));
%fprintf('length(step) = %d\n', length(step));
for i = 1: length(step)
    for j = 1: length(step)
        C = step(i);
        sigma = step(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        J(i, j) = mean(double(predictions ~= yval));
    end
end

min_C = 1;
min_sigma = 1;
for i = 1:length(step)
    for j = 1:length(step)
        if J(i, j) < J(min_C, min_sigma)
            min_C = i;
            min_sigma = j;
        end
    end
end

C = step(min_C);
sigma = step(min_sigma);




% =========================================================================

end
