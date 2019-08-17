clear;

%% step1: �������ݼ�
flower = load('flower.txt');
train_X = [flower(1:40, :);flower(51:90, :)];
train_Y = [ones(40, 1); zeros(40, 1)];
test_X = [flower(41:50, :);flower(91:end, :)];
test_Y = [ones(10, 1); zeros(10, 1)];

%% step2��ѧϰ��Ԥ��,Ԥ��test_y
output_perceptron = perceptron(train_X, train_Y, test_X);%��Ϊ���ȡ��㣬����ÿ�����з�������һ��һ��
output_knn = knn(train_X, train_Y, test_X); %�򵥲��ԣ�kȡ9Ч���ȽϺ�
output_logistic = logistic(train_X, train_Y, test_X);
output_entropy = entropy(train_X, train_Y, test_X);
output_tree = tree(train_X, train_Y, test_X);
output_nb = nb(train_X, train_Y, test_X); %���ݼ�Ϊ���������ݣ����ø�˹��Ҷ˹������

%% step3�����������׼ȷ���ٻ��ʵ�
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
    xlabel('����������');
    ylabel('ָ��ֵ');
end
function [ACC,PRE,REC] = evaluation(test_Y,output) %�����������ܣ�ACC����׼ȷ�ʡ�PRE��ȷ�ʡ�REC�ٻ���
    TP = 0;
    FN = 0;
    FP = 0;
    TN = 0;
    for i = 1:length(test_Y)
        if test_Y(i)==1
            if output(i)==1
                TP = TP + 1;%������Ԥ��Ϊ������
            else
                FN = FN + 1;%������Ԥ��Ϊ������
            end
        else
            if output(i)==1
                FP = FP + 1;%������Ԥ��Ϊ������
            else
                TN = TN + 1;%������Ԥ��Ϊ������
            end
        end
    end
    ACC = (TP+TN)/(TP+FN+FP+TN);
    PRE = TP/(TP+FP);
    REC = TP/(TP+FN);
end