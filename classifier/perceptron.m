function [ test_Y ] = perceptron( X, Y, test_X )
%% 初始化w,b,alpha,
w = [0,0,0,0];
b = 0;
alpha = 1;  % learning rate
sample = X;
for i = 1:length(Y)
    if Y(i)==1
        sign(i) = 1;
    else
        sign(i) = -1;
    end
end
sign = sign';
maxstep = 1000;
%%  更新 w,b
for i=1:maxstep
    [idx_misclass, counter] = class(sample, sign, w, b);%等式右边为输入，左边为输出，理解为函数调用
    %obj_struct = class(struct_array,'class_name',parent_array) 
    %idx_misclass:误分类序号索引，counter:
    if (counter~=0)%~=代表不等于
        R = unidrnd(counter);%产生离散均匀随机整数(一个),即随机选取起点训练
        %fprintf('%d\n',R);
        w = w + alpha * sample(idx_misclass(R),:) * sign(idx_misclass(R));
        b = b + alpha * sign(idx_misclass(R));   
    else
        break
    end
end
%%
for i=1:length(test_X)
    if(w*test_X(i,:)'+b >=0)
        test_Y(i) = 1;
    else
        test_Y(i) = 0;
    end
end
end

function [idx_misclass, counter] = class(sample, label, w, b)
    counter = 0;
    idx_misclass = [];
    for i=1:length(label)
        if (label(i)*(w*sample(i,:)'+b)<=0) %如果有误分类点，进行迭代
            idx_misclass = [idx_misclass i];
            %fprintf('%d\n',idx_misclass);
            counter = counter + 1;        
        end   
    end
end

