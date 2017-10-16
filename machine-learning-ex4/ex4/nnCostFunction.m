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

X = [ones(m, 1) X];
a = sigmoid(X * Theta1');
a = [ones(m, 1) a];
htheta = sigmoid(a * Theta2');

[m, n] = size(htheta);

one = ones(n, 1);

for i = 1 : m
    hone = htheta(i,:);
    
    yone = zeros(1, n);
    yone(y(i)) = 1;
    
    J = J + log(hone) * yone' + log(one' - hone) * (one - yone');
end

J = -J / m;

newTheta1 = Theta1(:, 2:end);
newTheta2 = Theta2(:, 2:end);
J = J + (sum(sum(newTheta1 .^ 2)) + sum(sum(newTheta2 .^ 2))) * lambda / (2 * m);










Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1 : m
    a1 = X(t,:)';
    z2 = Theta1 * a1;
    tmp2 = sigmoid(z2);
    a2 = [1; tmp2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    y1 = zeros(n, 1);
    y1(y(t)) = 1;
    
    delta3 = a3 - y1;
    
%     disp(size(Theta2'));
%     disp(size(delta3));
%     disp(size(z2));
    
    delta2 = (Theta2' * delta3) .* (a2 .* (ones(size(a2)) - a2));
    delta2 = delta2(2:end);
    
    Theta1_grad = Theta1_grad + delta2 * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';
    
end

factor1 = Theta1(:,2:end);
factor1 = [zeros(size(Theta1(:,1))) factor1];
Theta1_grad = Theta1_grad / m + factor1 * lambda / m;

factor2 = Theta2(:,2:end);
factor2 = [zeros(size(Theta2(:,1))) factor2];
Theta2_grad = Theta2_grad / m + factor2 * lambda / m;


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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
