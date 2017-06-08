function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
% data = load('ex3data1.mat');
% X = data(:, [1, 2]); y = data(:, 3);
% [m, n] = size(X);
% 
% % Add intercept term to x and X_test
%  X = [ones(m, 1) X];
% % 
% % % Initialize fitting parameters
% theta = zeros(n + 1, 1);
% lambda = 0.1;


%m = length(y); % number of training examples

% Initialize some useful values


% You need to return the following variables correctly 

% z=transpose(theta)*transpose(X);
% 
% hX=sigmoid(z);
% 
% J = ((-1/m)*(log(hX))*(y) + (log (1 - hX))*(1 - y))  + ( (lambda/2*m) * ((theta.^2)) );
% 
m = length(y); % number of training examples
%b=transpose(theta);
%theta = zeros(1,401);
%mytheta=zeros(1,3);
mahsa=sigmoid(X*theta);
costSum =sum( (y'*log(mahsa)) + (1 - y')*log (1 - mahsa));
theta(1)=0;
stheta=sum(theta.^2);
costT=sum((lambda/(2*m)).* stheta);
J = ((-1/m) * costSum) + costT;


if J==0
    %grad = (1/m) * ((mahsa - transpose(y))'*X);
    grad = (1/m) * ((mahsa - (y))'*X);
else 
   %grad = ((1/m) * ((mahsa - transpose(y))'*X)) + ((lambda/m) * transpose(theta)) ; 
    grad = ((1/m) * ((mahsa - (y))'*X)) + ((lambda/m) * transpose(theta)) ; 
   grad=grad';
end

% if J==0
%     grad = ((((1/m) .* (sum ((mahsa-transpose(y))*X)))));
% else 
% %     y=[y,[0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ]];
% %     y=[y,[0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ]];
% %     y=[y,[0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ]];
%    grad = ((((1/m) .* (sum ((mahsa-(y)')'*X))) )+ ((lambda/m).* theta)') ; 
% end

% grad = zeros(size(theta));
% 
% grad = ((1/m) * ((hX - transpose(y))*X)) + ((lambda/m) .* theta) ; 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================


end
