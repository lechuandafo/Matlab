function recomd_set = personalRank(train,testUserList)
    %% �������� �ɺ���
    % sample = [1 1 1;
    %          1 3 1;
    %          2 1 1;
    %          2 2 1;
    %          2 3 1;
    %          2 4 1;
    %          3 3 1;
    %          3 4 1];
    % G = get_rating(sample);
    %% ѵ�����ݲ��ڲ��Լ��ϲ���
    G = get_rating(train);
    max_step = 5;
    topk = 50;
    recomd_set = [];
    for i = 1:length(testUserList)
        fprintf("����Ϊ�� %d λ�û������Ƽ�\n�Ƽ�����Ŀ���Ϊ��\n",testUserList(i));
        testu =  strcat('a' , num2str(testUserList(i)));
        rec_testu = prank(G,0.85,testu,max_step,topk)
        recomd_set = [recomd_set;rec_testu];
    end
    %% ���д���ļ� ����ʱ��ǳ��ã��ڵ�ǰ�����²����Ҫ24Сʱ
    fid=fopen(['PRankrecomd_set.txt'],'w');%д���ļ�·��
    [r,c]=size(recomd_set);            % �õ����������������
    for i=1:r
        for j=1:c
            fprintf(fid,'%d\t',recomd_set(i,j));
        end
        fprintf(fid,'\r\n');
    end
    fclose(fid);
end

%% ���ֵ�ṹʵ�ֹ���user-item ����ͼ
function Graph = get_rating(train)
    M = max(train(:, 1)); % user number
    N = max(train(:, 2)); % movie number
    Graph = containers.Map();
    for i = 1:M %���������û���Ӧ����Ŀ
        u_rating = containers.Map();
        for j = 1:length(train)
            if(train(j,1) == i)
                u_id = strcat('a' , num2str(train(j,1))); %a�����û�
                i_id = strcat('b' , num2str(train(j,2))); %b������Ŀ               
                u_rating(i_id) = train(j,3);
            end
        end
        Graph(u_id) = u_rating;
    end
    for i = 1:N %����������Ŀ��Ӧ���û�
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
    visited = G(root).keys(); %�û����ʹ���item���Ƽ�ʱ�޳�
    for i = 1:G.Count()
        rank(char(keys(i))) = 0.0;
    end
    rank(root) = 1.0;
    %��ʼ����
    for k = 1:max_step
        tmp = containers.Map();
        %ȡ�ڵ�i�����ĳ���β�ڵ㼯��ri
        for i = 1:G.Count()
            tmp(char(keys(i))) = 0.0;
        end
        for s = 1:G.Count()
            i = char(keys(s));
            ri = G(char(keys(s)));
            ri_keys = ri.keys();
            for t = 1:ri.Count() %ȡ�ڵ�i�ĳ��ߵ�β�ڵ�j�Լ���E(i,j)��Ȩ��wij
                %i��j������һ����ߵ��׽ڵ㣬�����Ҫ����ͼ�ҵ�j����ߵ��׽ڵ㣬
                %����������̾��Ǵ˴���2��forѭ����һ�α�������һ������
                j = char(ri_keys(t));
                wij = ri(char(ri_keys(t)));
                tmp(j) = tmp(j) + alpha * double(rank(i))/double((wij * ri.Count));
            end
        end
        %����ÿ�����߶��Ǵ�root�ڵ���������root�ڵ��Ȩ����Ҫ����(1 - alpha)
        tmp(root) = tmp(root) + (1 - alpha);
        rank = tmp ;
        rank_keys = rank.keys();
    end
    result = [rank.keys();rank.values()]';%���н���Ӧ�ĸ���ֵ<Node,P>
    recomd = [];
    for i = 1:length(result)
        i_name = char(result(i,1));
        if(i_name(1)=='a') %��Ϊ���Ƽ���Ŀ�����԰��û�����ų�
        elseif (find(ismember(visited, i_name ))) %���ų����û����ʹ��Ľ��
        else
            recomd = [recomd;result(i,:)]; %ʣ����Ƽ����
        end
    end
    recomd = sortrows(recomd,2,'descend'); %������ֵ��������
    rec_topk = zeros(topk,1)';%ȡtopk
    for i = 1:topk
        if(i>length(recomd)) %�൱��topk > M+Nʱ�����
            break;
        end
        item = char(recomd(i,1));
        item(1) = [];
        rec_topk(i) = str2num(item);
    end
end