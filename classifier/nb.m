function [ test_Y ] = nb( X, Y, test_X )
    %% 计算每个分类的样本的概率
    labelProbability = tabulate(Y);
    %P_yi,计算P(yi)
    P_y1=labelProbability(1,3)/100;
    P_y2=labelProbability(2,3)/100;
    %% 计算每个属性(列)的均值与方差
    X1 = [];
    X2 = [];
    for i = 1:length(X)
        if(Y(i)==1)
            X1 = [X1;X(i,:)];
        else
            X2 = [X2;X(i,:)];
        end
    end
    %% 计算每个类的均值和方差
    x1_mean = mean(X1,1);
    x1_std = std(X1,0,1);
    x2_mean = mean(X2,1);
    x2_std = std(X2,0,1);

    test_Y = [];
    for i = 1:length(test_X)
        p1 = 1; %归属第一个类的概率
        p2 = 1; %归属第二个类的概率
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
    %% 定义高斯(正太)分布函数
    function p = gaussian(x,mean,std)
        p = 1/sqrt(2*pi*std*std)*exp(-(x-mean)*(x-mean)/(2*std));
    end
end