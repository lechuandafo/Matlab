function [ test_Y ] = tree( X, Y, test_X )
    global tree;
    tree = [];
    disp("The Tree is :");
    build_tree(X,Y);
%     disp(tree);
    S = regexp(tree, '\s+', 'split');
    test_Y = [];
    var = 0;
    condi = 0;
    for i = 1:length(test_X)
        tx = test_X(i,:);
        for j=1:length(S)
            if contains(S(j),'node')
                 var = tx(str2num(char(S(j+1))));
                 condi = str2num(char(S(j+2)));
                 if var < condi
                     if contains(S(j+3),'leaf')
                         ty = char(S((j+3)));
                         ty = ty(6);
                         test_Y = [test_Y,str2num(ty)];
                         break
                     else
                         j = j + 3;
                     end
                 else
                     if contains(S(j+4),'leaf')
                         ty = char(S((j+4)));
                         ty = ty(6);
                         test_Y = [test_Y,str2num(ty)];
                         break
                     else
                         j = j + 4;
                     end
                 end
            end
        end
    end
end

function build_tree(x, y, L, level, parent_y, sig, p_value)
    % 自编的用于构建决策树的简单程序，适用于属性为二元属性，二分类情况。（也可对程序进行修改以适用连续属性）。
    % 输入：
    % x：数值矩阵，样本属性记录（每一行为一个样本）
    % y：数值向量，样本对应的labels
    % 其它参数调用时可以忽略，在递归时起作用。
    % 输出：打印决策树。
    global tree;
    if nargin == 2
       level = 0; 
       parent_y = -1;
       L = 1:size(x, 2);
       sig = -1;
       p_value = [];
%        bin_f = zeros(size(x, 2), 1);
%        for k=1:size(x, 2)
%            if length(unique(x(:,k))) == 2
%               bin_f(k) = 1; 
%            end
%        end
    end
    class = [0, 1];
    [r, label] = is_leaf(x, y, parent_y); % 判断是否是叶子节点
    if r   
        if sig ==-1
            disp([repmat('     ', 1, level), 'leaf (', num2str(label), ')'])
            tree = [tree,[' leaf(', num2str(label), ')']];
        elseif sig ==0
            disp([repmat('     ', 1, level), '<', num2str(p_value),' leaf (', num2str(label), ')']);
            tree = [tree,[' leaf(', num2str(label), ')']];
        else
            disp([repmat('     ', 1, level), '>', num2str(p_value),' leaf (', num2str(label), ')']);
            tree = [tree,[' leaf(', num2str(label), ')']];
        end
    else
        [ind, value, i_] = find_best_test(x, y, L); % 找出最佳的测试值
%         
%         if ind ==1
%            keyboard; 
%         end

        [x1, y1, x2, y2] = split_(x, y, i_, value); % 实施划分
        if sig ==-1
            disp([repmat('     ', 1, level), 'node (', num2str(ind), ', ', num2str(value), ')']);
            tree = [tree,[' node ', num2str(ind), ' ',num2str(value)]];
        elseif sig ==0
            disp([repmat('     ', 1, level), '<', num2str(p_value),' node (', num2str(ind), ', ', num2str(value), ')']);
            tree = [tree,[' node ',num2str(ind), ' ', num2str(value)]];
        else
            disp([repmat('     ', 1, level), '>', num2str(p_value),' node (', num2str(ind), ', ', num2str(value), ')']);
            tree = [tree,[' node ',num2str(ind), ' ', num2str(value)]];
        end
%         if bin_f(i_) == 1
            x1(:,i_) = []; 
            x2(:,i_) = [];
            L(:,i_) = [];
%             bin_f(i_) = [];
%         end
        build_tree(x1, y1, L, level+1, y, 0, value); % 递归调用
        build_tree(x2, y2, L, level+1, y, 1, value);
    end

    function [ind, value, i_] = find_best_test(xx, yy, LL) % 子函数：找出最佳测试值（可以对连续属性适用）
        imp_min = inf;
        i_ = 1;
        ind = LL(i_);
        for i=1:size(xx,2);
            if length(unique(xx(:,i))) ==1
                continue;
            end
%            [xx_sorted, ii] = sortrows(xx, i); 
%            yy_sorted = yy(ii, :);
           vv = unique(xx(:,i));
           imp_min_i = inf;
           best_point = mean([vv(1), vv(2)]);
           value = best_point;
           for j = 1:length(vv)-1
               point = mean([vv(j), vv(j+1)]);               
               [xx1, yy1, xx2, yy2] = split_(xx, yy, i, point);
               imp = calc_imp(yy1, yy2);
               if imp<imp_min_i
                   best_point = point;
                   imp_min_i = imp;
               end
           end
           if imp_min_i < imp_min
              value = best_point;
              imp_min = imp_min_i;
              i_ = i;
              ind = LL(i_);
           end
        end
    end

    function imp = calc_imp(y1, y2) % 子函数：计算熵
        p11 = sum(y1==class(1))/length(y1);
        p12 = sum(y1==class(2))/length(y1);
        p21 = sum(y2==class(1))/length(y2);
        p22 = sum(y2==class(2))/length(y2);
        if p11==0
            t11 = 0;
        else
           t11 = p11*log2(p11); 
        end
        if p12==0
            t12 = 0;
        else
           t12 = p12*log2(p12); 
        end
        if p21==0
            t21 = 0;
        else
           t21 = p21*log2(p21); 
        end
        if p22==0
            t22 = 0;
        else
           t22 = p22*log2(p22); 
        end

        imp = -t11-t12-t21-t22;
    end

    function [x1, y1, x2, y2] = split_(x, y, i, point) % 子函数：实施划分
       index = (x(:,i)<point);
       x1 = x(index,:);
       y1 = y(index,:);
       x2 = x(~index,:);
       y2 = y(~index,:);
    end

    function [r, label] = is_leaf(xx, yy, parent_yy) % 子函数：判断是否是叶子节点
        if isempty(xx)
            r = true;
            label = mode(parent_yy);
        elseif length(unique(yy)) == 1
            r = true;
            label = unique(yy);
        else
            t = xx - repmat(xx(1,:),size(xx, 1), 1);
            if all(all(t ==0))
                r = true;
                label = mode(yy);
            else
                r = false;
                label = [];
            end
        end
    end
end