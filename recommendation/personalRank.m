function recomd_set = personalRank(train,testUserList)
    %% 测试样例 可忽略
    % sample = [1 1 1;
    %          1 3 1;
    %          2 1 1;
    %          2 2 1;
    %          2 3 1;
    %          2 4 1;
    %          3 3 1;
    %          3 4 1];
    % G = get_rating(sample);
    %% 训练数据并在测试集上测试
    G = get_rating(train);
    max_step = 5;
    topk = 50;
    recomd_set = [];
    for i = 1:length(testUserList)
        fprintf("正在为第 %d 位用户进行推荐\n推荐的项目编号为：\n",testUserList(i));
        testu =  strcat('a' , num2str(testUserList(i)));
        rec_testu = prank(G,0.85,testu,max_step,topk)
        recomd_set = [recomd_set;rec_testu];
    end
    %% 结果写入文件 运行时间非常久，在当前参数下差不多需要24小时
    fid=fopen(['PRankrecomd_set.txt'],'w');%写入文件路径
    [r,c]=size(recomd_set);            % 得到矩阵的行数和列数
    for i=1:r
        for j=1:c
            fprintf(fid,'%d\t',recomd_set(i,j));
        end
        fprintf(fid,'\r\n');
    end
    fclose(fid);
end

%% 用字典结构实现构建user-item 二部图
function Graph = get_rating(train)
    M = max(train(:, 1)); % user number
    N = max(train(:, 2)); % movie number
    Graph = containers.Map();
    for i = 1:M %构建所有用户对应的项目
        u_rating = containers.Map();
        for j = 1:length(train)
            if(train(j,1) == i)
                u_id = strcat('a' , num2str(train(j,1))); %a代表用户
                i_id = strcat('b' , num2str(train(j,2))); %b代表项目               
                u_rating(i_id) = train(j,3);
            end
        end
        Graph(u_id) = u_rating;
    end
    for i = 1:N %构建所有项目对应的用户
        i_rating = containers.Map();
        for j = 1:length(train)
            if(train(j,2) == i)
                u_id = strcat('a' , num2str(train(j,1)));
                i_id = strcat('b' , num2str(train(j,2)));
                i_rating(u_id) = train(j,3);
            end
        end
        Graph(i_id) = i_rating;
    end    
end

%% Pernoal Rank
function rec_topk = prank (G,alpha,root,max_step,topk)
    rank = containers.Map();
    keys = G.keys();
    visited = G(root).keys(); %用户访问过的item，推荐时剔除
    for i = 1:G.Count()
        rank(char(keys(i))) = 0.0;
    end
    rank(root) = 1.0;
    %开始迭代
    for k = 1:max_step
        tmp = containers.Map();
        %取节点i和它的出边尾节点集合ri
        for i = 1:G.Count()
            tmp(char(keys(i))) = 0.0;
        end
        for s = 1:G.Count()
            i = char(keys(s));
            ri = G(char(keys(s)));
            ri_keys = ri.keys();
            for t = 1:ri.Count() %取节点i的出边的尾节点j以及边E(i,j)的权重wij
                %i是j的其中一条入边的首节点，因此需要遍历图找到j的入边的首节点，
                %这个遍历过程就是此处的2层for循环，一次遍历就是一次游走
                j = char(ri_keys(t));
                wij = ri(char(ri_keys(t)));
                tmp(j) = tmp(j) + alpha * double(rank(i))/double((wij * ri.Count));
            end
        end
        %我们每次游走都是从root节点出发，因此root节点的权重需要加上(1 - alpha)
        tmp(root) = tmp(root) + (1 - alpha);
        rank = tmp ;
        rank_keys = rank.keys();
    end
    result = [rank.keys();rank.values()]';%所有结点对应的概率值<Node,P>
    recomd = [];
    for i = 1:length(result)
        i_name = char(result(i,1));
        if(i_name(1)=='a') %因为是推荐项目，所以把用户结点排除
        elseif (find(ismember(visited, i_name ))) %再排除掉用户访问过的结点
        else
            recomd = [recomd;result(i,:)]; %剩余的推荐结果
        end
    end
    recomd = sortrows(recomd,2,'descend'); %按概率值降序排列
    rec_topk = zeros(topk,1)';%取topk
    for i = 1:topk
        if(i>length(recomd)) %相当于topk > M+N时的情况
            break;
        end
        item = char(recomd(i,1));
        item(1) = [];
        rec_topk(i) = str2num(item);
    end
end