function [ test_Y ] = nb( X, Y, test_X )
    %% ����ÿ������������ĸ���
    labelProbability = tabulate(Y);
    %P_yi,����P(yi)
    P_y1=labelProbability(1,3)/100;
    P_y2=labelProbability(2,3)/100;
    %% ����ÿ������(��)�ľ�ֵ�뷽��
    X1 = [];
    X2 = [];
    for i = 1:length(X)
        if(Y(i)==1)
            X1 = [X1;X(i,:)];
        else
            X2 = [X2;X(i,:)];
        end
    end
    %% ����ÿ����ľ�ֵ�ͷ���
    x1_mean = mean(X1,1);
    x1_std = std(X1,0,1);
    x2_mean = mean(X2,1);
    x2_std = std(X2,0,1);

    test_Y = [];
    for i = 1:length(test_X)
        p1 = 1; %������һ����ĸ���
        p2 = 1; %�����ڶ�����ĸ���
        for j = 1:length(test_X(1,:))
            p1 = p1 * gaussian(test_X(i,j),x1_mean(j),x1_std(j)); %p(xj|y)
            p2 = p2 * gaussian(test_X(i,j),x2_mean(j),x2_std(j));
        end
        p1 = p1 * P_y1;
        p2 = p2 * P_y2;
        if(p1>p2)
            test_Y = [test_Y,1];
        else
            test_Y = [test_Y,0];
        end
    end
    %% �����˹(��̫)�ֲ�����
    function p = gaussian(x,mean,std)
        p = 1/sqrt(2*pi*std*std)*exp(-(x-mean)*(x-mean)/(2*std));
    end
end