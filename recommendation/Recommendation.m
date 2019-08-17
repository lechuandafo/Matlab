clear;
movie = load('movie.txt');
%% Ԥ��������ݼ��ָ�
% get user-item-rating, ordered by time
movie = sortrows(sortrows(movie, 4), 1);
[~, ~, userlist] = unique(movie(:, 1));
[~, ~, itemlist] = unique(movie(:, 2));
movie(:, 1) = userlist;
movie(:, 2) = itemlist;
clear userlist itemlist
M = max(movie(:, 1)); % user number
N = max(movie(:, 2)); % movie number
[~, userlastpos] = unique(movie(:, 1));
userlastpos = [userlastpos(2:end)-1; size(movie, 1)];
NUM_TEST = 10;
isTest = zeros(size(movie, 1), 1);
for u = 1:M
    thisend = userlastpos(u);
    isTest((thisend-9):thisend) = 1;
end

train = movie(isTest == 0, :);
test = movie(isTest == 1, :);
% remove the moives with less than 3 stars in the test set
test(test(:, 3) < 3, :) = [];
test(:, 3:4) = [];
testUserList = 1:M;
%% ѵ����Ԥ��
% ÿ���㷨Ԥ��ÿ�������û�50������ϲ�����������ῴ����û�����ĵ�Ӱ������Ϊcell��ʽ���Լ�ȷ��
if ~exist('HMMrecomd_set.txt','file')
    disp('����ִ��HMM�Ƽ�');
    output_HMM = HMM(train,testUserList);
elseif ~exist('PRankrecomd_set.txt','file')
	disp('����ִ��personalRank�Ƽ�');
    output_personalRank = personalRank(train,testUserList);
else
    %% �����Ƚ�
    HMM_recomd = load('HMMrecomd_set.txt');
    PR_recomd = load('PRankrecomd_set.txt');
    test_visit = {};
    for i = 1:length(testUserList)
        u_visit = [];
        for j = 1:length(test)
            if (i == test(j,1))
                u_visit = [u_visit,test(j,2)];
            end
        end
        test_visit{i} = u_visit;
    end
    [HMM_PRE,HMM_REC] = eva_recomd(test_visit,HMM_recomd)
    [PR_PRE,PRE_REC] = eva_recomd(test_visit,PR_recomd)
    result = [HMM_PRE,HMM_REC;PR_PRE,PRE_REC];
    bar(result);
    grid on;
    legend('PRE','REC');
    set(gca,'XTickLabel',{'HMM','personalRank'});
    xlabel('�Ƽ��㷨����');
    ylabel('ָ��ֵ');
end
    
function [PRE,REC] = eva_recomd(test_visit,recomd)
    REC = 0;
    true_count = 0;
    no_test_count = 0;

    for i = 1:length(recomd)
        for j = 1:length(recomd(i,:))
            if(isempty(test_visit{i})) %��Щ�û���test��û�����ݣ���82��125��129��181��515��804
    %             disp(test_visit{i});
    %             fprintf("%d\n",i);
                no_test_count = no_test_count + 1;
                continue;
            else
                rec_count = 0;
                for k = 1:length(test_visit{i})
                    t_visit = test_visit{i};
                    if(t_visit(k) ==  recomd(i,j))
                        true_count = true_count + 1;
                        rec_count = rec_count + 1;
                    end
                end
                if rec_count > length(test_visit{i})
                    fprintf("%d, %d",rec_count,length(test_visit{i}));
                end
                REC = REC + double(rec_count) / double(length(test_visit{i}));
            end
        end
    end
    PRE = double(true_count) / double((length(recomd)-no_test_count)*length(recomd(i,:)));
    REC = REC / double(length(recomd)-no_test_count);
end
