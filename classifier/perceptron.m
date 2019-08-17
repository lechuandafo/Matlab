function [ test_Y ] = perceptron( X, Y, test_X )
%% ��ʼ��w,b,alpha,
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
%%  ���� w,b
for i=1:maxstep
    [idx_misclass, counter] = class(sample, sign, w, b);%��ʽ�ұ�Ϊ���룬���Ϊ��������Ϊ��������
    %obj_struct = class(struct_array,'class_name',parent_array) 
    %idx_misclass:��������������counter:
    if (counter~=0)%~=��������
        R = unidrnd(counter);%������ɢ�����������(һ��),�����ѡȡ���ѵ��
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
        if (label(i)*(w*sample(i,:)'+b)<=0) %����������㣬���е���
            idx_misclass = [idx_misclass i];
            %fprintf('%d\n',idx_misclass);
            counter = counter + 1;        
        end   
    end
end

