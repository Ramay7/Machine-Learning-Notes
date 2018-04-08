function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
col = size(X, 2);
X = [ones(m, 1) X];
X1 = sigmoid(X * Theta1');
X1 = [ones(m, 1) X1];
result = sigmoid(X1 * Theta2');

maxCol = max(result, [], 2);

for i = 1 : m
    for j = 1 : num_labels
        if result(i, j) == maxCol(i)
            p(i) = j;
            break
        end
    end
end

%fprintf('size(X) = (%d, %d)\n', size(X));
%fprintf('size(Theta1) = (%d, %d)\n', size(Theta1));
%fprintf('size(Theta2) = (%d, %d)\n', size(Theta2));



% =========================================================================


end
