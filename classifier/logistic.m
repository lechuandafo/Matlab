function [ test_Y ] = logistic( X, Y, test_X )
m = size(X,1);
n = size(X,2);
initial_theta = zeros(size(X, 2), 1);

% 正则化参数，不对theta0惩罚，在costfunction中体现。避免过拟合。
% lambda值越大，theta值越小。
lambda = 1;

[cost, grad] = costFunction(initial_theta, X, Y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
% GradObj-on 表示costFunction同时返回cost和grad
options = optimset('GradObj', 'on', 'MaxIter', 400);

% f = @(t)(...t...) 创建了一个函数句柄。
% f为函数名称，t为输入变量，后面为执行语句。
f = @(t)(costFunction(t, X, Y, lambda));
[theta, J, exit_flag] = fminunc(f, initial_theta, options);
test_Y = sigmoid(test_X * theta) >= 0.5;
test_Y = test_Y';
end

function [J, grad] = costFunction(theta, X, Y, lambda)
    m = length(Y);
    grad = zeros(size(theta));

    J = 1 / m * sum( -Y .* log(sigmoid(X * theta)) - (1 - Y) .* log(1 - sigmoid(X * theta)) ) + lambda / (2 * m) * sum( theta(2 : size(theta), :) .^ 2 );

    for j = 1 : size(theta)
        if j == 1
            grad(j) = 1 / m * sum( (sigmoid(X * theta) - Y)' * X(:, j) );
        else
            grad(j) = 1 / m * sum( (sigmoid(X * theta) - Y)' * X(:, j) ) + lambda / m * theta(j);
        end
    end

end

function g = sigmoid(z)
    g = zeros(size(z));
    g = 1 ./ (1 + exp(-z));
end






