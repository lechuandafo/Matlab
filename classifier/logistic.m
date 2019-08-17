function [ test_Y ] = logistic( X, Y, test_X )
m = size(X,1);
n = size(X,2);
initial_theta = zeros(size(X, 2), 1);

% ���򻯲���������theta0�ͷ�����costfunction�����֡��������ϡ�
% lambdaֵԽ��thetaֵԽС��
lambda = 1;

[cost, grad] = costFunction(initial_theta, X, Y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
% GradObj-on ��ʾcostFunctionͬʱ����cost��grad
options = optimset('GradObj', 'on', 'MaxIter', 400);

% f = @(t)(...t...) ������һ�����������
% fΪ�������ƣ�tΪ�������������Ϊִ����䡣
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






