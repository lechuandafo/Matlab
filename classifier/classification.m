clear;

%% step1: 划分数据集
flower = load('flower.txt');
train_X = [flower(1:40, :);flower(51:90, :)];
train_Y = [ones(40, 1); zeros(40, 1)];
test_X = [flower(41:50, :);flower(91:end, :)];
test_Y = [ones(10, 1); zeros(10, 1)];

%% step2：学习和预测,预测test_y
output_perceptron = perceptron(train_X, train_Y, test_X);%因为随机取起点，所以每次运行分类结果不一定一样
output_knn = knn(train_X, train_Y, test_X); %简单测试，k取9效果比较好
output_logistic = logistic(train_X, train_Y, test_X);
output_entropy = entropy(train_X, train_Y, test_X);
output_tree = tree(train_X, train_Y, test_X);
output_nb = nb(train_X, train_Y, test_X); %数据集为连续型数据，需用高斯贝叶斯分类器

%% step3：分析结果，准确率召回率等
output_set = [output_perceptron;output_knn;output_logistic;output_entropy;output_tree;output_nb];
compare(test_Y,output_set);
function compare(test_Y,output)
    result = [];
    for i = 1:length(output(:,1))
        [ACC,PRE,REC] = evaluation(test_Y,output(i,:));
        result = [result;[ACC,PRE,REC]];
    end
    bar(result);
    grid on;
    legend('ACC','PRE','REC');
    set(gca,'XTickLabel',{'perceptron','knn','logistic','entropy','tree','nb'});
    xlabel('分类器种类');
    ylabel('指标值');
end
function [ACC,PRE,REC] = evaluation(test_Y,output) %评估分类性能，ACC分类准确率、PRE精确率、REC召回率
    TP = 0;
    FN = 0;
    FP = 0;
    TN = 0;
    for i = 1:length(test_Y)
        if test_Y(i)==1
            if output(i)==1
                TP = TP + 1;%将正类预测为正类数
            else
                FN = FN + 1;%将正类预测为负类数
            end
        else
            if output(i)==1
                FP = FP + 1;%将负类预测为正类数
            else
                TN = TN + 1;%将负类预测为负类数
            end
        end
    end
    ACC = (TP+TN)/(TP+FN+FP+TN);
    PRE = TP/(TP+FP);
    REC = TP/(TP+FN);
end