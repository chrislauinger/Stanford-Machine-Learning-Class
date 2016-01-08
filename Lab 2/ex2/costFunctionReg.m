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
[old_J, old_grad] = costFunction(theta,X,y);
reg_term = 0;
for j = 2:length(theta)
    reg_term = reg_term + theta(j)*theta(j);
end
J = old_J + (lambda / (2*m))*reg_term;

grad(1) = old_grad(1);
for j = 2:length(theta)
    grad(j) = old_grad(j) + (lambda / m) * theta(j);
end

% =============================================================

end
