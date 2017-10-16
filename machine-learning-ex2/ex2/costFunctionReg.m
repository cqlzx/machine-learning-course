function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
% J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
htheta = sigmoid(X * theta);
one = ones(m, 1);
thetawithoutfirst = theta(2:n,:);

part1 = (-1 / m) * (y' * log(htheta) + (one - y)' * log(one - htheta));
part2 = (lambda / (2 * m)) * sum(thetawithoutfirst .^ 2);

J = part1 + part2;

grad(1) = (1 / m) * (X(:,1)' * (htheta - y));
for i = 2 : n
    grad(i) = (1 / m) * (X(:,i)' * (htheta - y)) + (lambda / m) * theta(i);
end



% =============================================================

end
