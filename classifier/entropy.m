function [ test_Y ] = entropy( X, Y, test_X )
    %% 变量定义
    maxstep=10;
    w = [];
    labels = []; %类别的种类
    fea_list = []; %特征集合
    px = containers.Map(); %经验边缘分布概率
    pxy = containers.Map(); %经验联合分布概率,由于特征函数为取值为0，1的二值函数，所以等同于特征的经验期望值
    exp_fea = containers.Map(); %每个特征在数据集上的期望 
    N = 0; % 样本总量
    M =0; % 某个训练样本包含特征的总数，这里假设每个样本的M值相同，即M为常数。其倒数类似于学习率
    n_fea = 0; % 特征函数的总数
    fit(X,Y);
    test_Y = [];
    for q = 1:length(test_X) %测试集预测
        p_Y = predict(test_X(q));
        test_Y = [test_Y,p_Y];
    end
    %% 变量初始化
    function init_param(X,Y)
        N = length(X);
        labels = unique(Y);
        
        %初始化px pxy
        for i = 1:length(X)
            px(num2str(X(i,:))) = 0;
            pxy(num2str([X(i,:),Y(i)])) = 0;
        end
        
        %初始化exp_fea
        for i = 1:length(X)
            for j = 1:length(X(i,:))
                fea = [j,X(i,j),Y(i)];
                exp_fea(num2str(fea)) = 0;
            end
        end
        fea_func(X,Y);
        n_fea = length(fea_list);
        w = zeros(n_fea,1);
        f_exp_fea(X,Y); 
    end
    %% 构造特征函数
    function fea_func(X,Y)
        for i = 1:length(X)
            px(num2str(X(i,:))) = px(num2str(X(i,:))) + 1.0/double(N);
            pxy(num2str([X(i,:),Y(i)])) = pxy(num2str([X(i,:),Y(i)]))+ 1.0/double(N);
            for j = 1:length(X(1,:))
                key = [j,X(i,j),Y(i)];
                fea_list = [fea_list;key];
            end
        end
        fea_list = unique(fea_list,'rows');
        M = length(X(i,:));
    end
    function f_exp_fea(X,Y)
        for i = 1:length(X)
            for j = 1:length(X(i,:))
                fea = [j,X(i,j),Y(i)];
                exp_fea(num2str(fea)) = exp_fea(num2str(fea)) + pxy(num2str([X(i,:),Y(i)]));
            end
        end
    end
    %% 当前w下的条件分布概率,输入向量X和y的条件概率
    function py_X =f_py_X(X)
        py_X = containers.Map();
        for i = 1:length(labels)
            py_X(num2str(labels(i))) = 0.0;
        end
        for i = 1:length(labels)
            s = 0;
            for j = 1:length(X)
                tmp_fea = [j,X(j),labels(i)];
                if(ismember(tmp_fea,fea_list,'rows'))
                    s = s + w(ismember(tmp_fea,fea_list,'rows'));
                end
            end
            py_X(num2str(labels(i))) = exp(s);
        end
        normalizer = 0;
        k_names = py_X.keys();
        for i = 1:py_X.Count
            normalizer = normalizer + py_X(char(k_names(i)));
        end
        for i = 1:py_X.Count
            py_X(char(k_names(i))) = double(py_X(char(k_names(i))))/double(normalizer);
        end
    end
    %% 基于当前模型，获取每个特征估计期望
    function est_fea = f_est_fea(X,Y)
        est_fea = containers.Map();
        for i = 1:length(X)
            for j = 1:length(X(i,:))
                est_fea(num2str([j,X(i,j),Y(i)])) = 0;
            end
        end
        for i = 1:length(X)
            py_x = f_py_X(X);
            py_x = py_x(num2str(Y(i)));
            for j = 1:length(X(i,:))
                est_fea(num2str([j,X(i,j),Y(i)])) = est_fea(num2str([j,X(i,j),Y(i)])) + px(num2str(X(i,:)))*py_x;
            end
        end
    end
    %% GIS算法更新delta
    function delta = GIS()
        est_fea = f_est_fea(X,Y);
        delta = zeros(n_fea,1);
        for i = 1:n_fea
            try
                delta(i) = 1 / double(M*log(exp_fea(num2str(fea_list(i,:)))))/double(est_fea(num2str(fea_list(i,:))));
            catch
                continue;
            end
        end
        delta = double(delta) / double(sum(delta));
    end
    %% 训练，迭代更新wi
    function fit(X,Y)
        init_param(X,Y);
        i = 0;
        while( i < maxstep)
            i = i + 1;
            delta = GIS();
            w = w + delta;
        end
    end
    %% 输入x(数组)，返回条件概率最大的标签
    function best_label = predict(X)
        py_x = f_py_X(X);
        keys = py_x.keys();
        values = py_x.values();
        max = 0;
        best_label = -1;
        for i = 1:py_x.Count
            if(py_x(char(keys(i))) > max)
                max = py_x(char(keys(i)));
                best_label = str2num(char(keys(i)));
            end
        end
    end
end