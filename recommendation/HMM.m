function output = HMM(train,testUserList)
    O = max(train(:, 2)); % 观测类别个数
    Q = 50; %状态类别个数
    o_set = get_oset(train);
    topk = 50;
    output = HMMrec(o_set,testUserList,O,Q,topk);
end

%% 获取训练集中所有用户历史交互序列
function o_set = get_oset(train)
    M = length(unique(train(:,1))); %训练集中用户个数，要保证用户编号连续
    user = train(:,1);
    item = train(:,2);
    cur_user = 1;
    o_set = {};%这里一开始不知道Matlab的字典结构Map,于是用元胞数组实现，比较麻烦
    o_list = []; %用户历史记录，作为观测序列
    count = 0;
    for i = 1:length(train)
        if(user(i)==cur_user)
            o_list = [o_list,item(i)];
        else
            if(user(i) == M) %最后一个用户补上
                count = count + 1;
                if(count == 1)
                    o_set = [o_set,o_list];
                    o_list = [];
                else
                    o_list = [o_list,item(i)]; 
                    if(i == length(train))
                        o_set = [o_set,o_list];
                        o_list = [];
                    end
                end
            else
                o_set = [o_set,o_list];
                o_list = [];
                cur_user = cur_user + 1;                
            end
        end
    end
end

%% 推荐
function recomd_set = HMMrec(o_set,testUserList,O,Q,topk)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ①定义一个HMM并训练这个HMM。
    % ②用一组观察值测试这个HMM，计算该组观察值域HMM的匹配度。
    % o_set: 所有用户观测序列集合    
    % O：观察状态数 应为item类别个数
    % Q：HMM状态数 自己定的参数，可理解为电影类别或者用户偏好类别？
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %训练的数据集,每一行数据就是一组训练的观察值
    data= o_set;

    % initial guess of parameters
    % 初始化参数
    prior1 = normalise(rand(Q,1));
    transmat1 = mk_stochastic(rand(Q,Q));
    obsmat1 = mk_stochastic(rand(Q,O));

    % improve guess of parameters using EM
    % 用data数据集训练参数矩阵形成新的HMM模型
    [LL, prior2, transmat2, obsmat2] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', size(data,1));
    % 训练后那行观察值与HMM匹配度
    LL;
    % 训练后的初始概率分布 即pi
    prior2;
    % 训练后的状态转移概率矩阵 即 A
    transmat2;
    % 观察值概率矩阵 即 B
    obsmat2;
    
    result = [];
    % use model to compute log 
    itemlist = [];
    for i = 1:O
        itemlist = [itemlist,1]; %项目列表
    end
    recomd_set = [];
    for i = 1:length(testUserList) %训练时用全部数据构造一条HMM，测试时只给testUserlist的用户推荐
        posilist = itemlist;
        for j = 1:length(o_set{testUserList(i)})
            posilist(o_set{testUserList(i)}(j)) = 0; %用户曾经访问过的item置为0
        end
        for k = 1:length(posilist)
            if(posilist(k)~=0) %如果不是用户访问过的item，则计算其添加到原序列后新序列的概率
                pred_list = [o_set{testUserList(i)},k]; %原序列加上第k个item构成新序列
                loglik = dhmm_logprob(pred_list, prior2, transmat2, obsmat2); %计算对数似然概率
                result = [result;[k,loglik]];
            end
        end
        result = sortrows(result,2,'descend'); %按对数似然概率降序排列
        recomd = [];
        for s = 1:topk
            recomd = [recomd,result(s,1)]; %取topK
        end
        fprintf("第%d位用户的top%d推荐结果为：\n",testUserList(i),topk);
        disp(recomd);
        recomd_set = [recomd_set;recomd];
    end
    fid=fopen(['HMMrecomd_set.txt'],'w');%写入文件路径
    [r,c]=size(recomd_set);            % 得到矩阵的行数和列数
     for i=1:r
      for j=1:c
      fprintf(fid,'%d\t',recomd_set(i,j));
      end
      fprintf(fid,'\r\n');
     end
    fclose(fid);
end  
    
%%%%%%%%%%%%%%%%%%%%%%%以下为HMM工具箱%%%%%%%%%%%%%%%%%%%%%%%%%
%% mk_stochastic
function CPT = mk_stochastic(CPT)
% MK_STOCHASTIC Make a matrix stochastic, i.e., the sum over the last dimension is 1.
% T = mk_stochastic(T)
%
% If T is a vector, it will be normalized.
% If T is a matrix, each row will sum to 1.
% If T is e.g., a 3D array, then sum_k T(i,j,k) = 1 for all i,j.
    if isvec(CPT)
        CPT = normalise(CPT);
    else
      n = ndims(CPT);
      % Copy the normalizer plane for each i.
      normalizer = sum(CPT, n);
      normalizer = repmat(normalizer, [ones(1,n-1) size(CPT,n)]);
      % Set zeros to 1 before dividing
      % This is valid since normalizer(i) = 0 iff CPT(i) = 0
      normalizer = normalizer + (normalizer==0);
      CPT = CPT ./ normalizer;
    end
end

%%%%%%%
function p = isvec(v)
    s=size(v);
    if ndims(v)<=2 & (s(1) == 1 | s(2) == 1)
      p = 1;
    else
      p = 0;
    end
end


%% dhmm_em
function [LL, prior, transmat, obsmat, nrIterations] = ...
   dhmm_em(data, prior, transmat, obsmat, varargin)
% LEARN_DHMM Find the ML/MAP parameters of an HMM with discrete outputs using EM.
% [ll_trace, prior, transmat, obsmat, iterNr] = learn_dhmm(data, prior0, transmat0, obsmat0, ...)
%
% Notation: Q(t) = hidden state, Y(t) = observation
%
% INPUTS:
% data{ex} or data(ex,:) if all sequences have the same length
% prior(i)
% transmat(i,j)
% obsmat(i,o)
%
% Optional parameters may be passed as 'param_name', param_value pairs.
% Parameter names are shown below; default values in [] - if none, argument is mandatory.
%
% 'max_iter' - max number of EM iterations [10]
% 'thresh' - convergence threshold [1e-4]
% 'verbose' - if 1, print out loglik at every iteration [1]
% 'obs_prior_weight' - weight to apply to uniform dirichlet prior on observation matrix [0]
%
% To clamp some of the parameters, so learning does not change them:
% 'adj_prior' - if 0, do not change prior [1]
% 'adj_trans' - if 0, do not change transmat [1]
% 'adj_obs' - if 0, do not change obsmat [1]
%
% Modified by Herbert Jaeger so xi are not computed individually
% but only their sum (over time) as xi_summed; this is the only way how they are used
% and it saves a lot of memory.

    [max_iter, thresh, verbose, obs_prior_weight, adj_prior, adj_trans, adj_obs] = ...
       process_options(varargin, 'max_iter', 10, 'thresh', 1e-4, 'verbose', 1, ...
                       'obs_prior_weight', 0, 'adj_prior', 1, 'adj_trans', 1, 'adj_obs', 1);

    previous_loglik = -inf;
    loglik = 0;
    converged = 0;
    num_iter = 1;
    LL = [];

    if ~iscell(data)
     data = num2cell(data, 2); % each row gets its own cell
    end

    while (num_iter <= max_iter) & ~converged
     % E step
     [loglik, exp_num_trans, exp_num_visits1, exp_num_emit] = ...
         compute_ess_dhmm(prior, transmat, obsmat, data, obs_prior_weight);

     % M step
     if adj_prior
       prior = normalise(exp_num_visits1);
     end
     if adj_trans & ~isempty(exp_num_trans)
       transmat = mk_stochastic(exp_num_trans);
     end
     if adj_obs
       obsmat = mk_stochastic(exp_num_emit);
     end

     if verbose, fprintf(1, 'iteration %d, sumloglik = %f\n', num_iter, loglik); end
     num_iter =  num_iter + 1;
     converged = em_converged(loglik, previous_loglik, thresh);
     previous_loglik = loglik;
     LL = [LL loglik];
    end
    nrIterations = num_iter - 1;
end

%%%%%%%%%%%%%%%%%%%%%%%

function [loglik, exp_num_trans, exp_num_visits1, exp_num_emit, exp_num_visitsT] = ...
   compute_ess_dhmm(startprob, transmat, obsmat, data, dirichlet)
% COMPUTE_ESS_DHMM Compute the Expected Sufficient Statistics for an HMM with discrete outputs
% function [loglik, exp_num_trans, exp_num_visits1, exp_num_emit, exp_num_visitsT] = ...
%    compute_ess_dhmm(startprob, transmat, obsmat, data, dirichlet)
%
% INPUTS:
% startprob(i)
% transmat(i,j)
% obsmat(i,o)
% data{seq}(t)
% dirichlet - weighting term for uniform dirichlet prior on expected emissions
%
% OUTPUTS:
% exp_num_trans(i,j) = sum_l sum_{t=2}^T Pr(X(t-1) = i, X(t) = j| Obs(l))
% exp_num_visits1(i) = sum_l Pr(X(1)=i | Obs(l))
% exp_num_visitsT(i) = sum_l Pr(X(T)=i | Obs(l))
% exp_num_emit(i,o) = sum_l sum_{t=1}^T Pr(X(t) = i, O(t)=o| Obs(l))
% where Obs(l) = O_1 .. O_T for sequence l.

    numex = length(data);
    [S O] = size(obsmat);
    exp_num_trans = zeros(S,S);
    exp_num_visits1 = zeros(S,1);
    exp_num_visitsT = zeros(S,1);
    exp_num_emit = dirichlet*ones(S,O);
    loglik = 0;

    for ex=1:numex
     obs = data{ex};
     T = length(obs);
     %obslik = eval_pdf_cond_multinomial(obs, obsmat);
     obslik = multinomial_prob(obs, obsmat);
     [alpha, beta, gamma, current_ll, xi_summed] = fwdback(startprob, transmat, obslik);

     loglik = loglik +  current_ll;
     exp_num_trans = exp_num_trans + xi_summed;
     exp_num_visits1 = exp_num_visits1 + gamma(:,1);
     exp_num_visitsT = exp_num_visitsT + gamma(:,T);
     % loop over whichever is shorter
     if T < O
       for t=1:T
         o = obs(t);
         exp_num_emit(:,o) = exp_num_emit(:,o) + gamma(:,t);
       end
     else
       for o=1:O
         ndx = find(obs==o);
         if ~isempty(ndx)
           exp_num_emit(:,o) = exp_num_emit(:,o) + sum(gamma(:, ndx), 2);
         end
       end
     end
    end
end

%% em_converged
function [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
% EM_CONVERGED Has EM converged?
% [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
%
% We have converged if
%   |f(t) - f(t-1)| / avg < threshold,
% where avg = (|f(t)| + |f(t-1)|)/2 and f is log lik.
% threshold defaults to 1e-4.

if nargin < 3
  threshold = 1e-4;
end

converged = 0;
decrease = 0;

if loglik - previous_loglik < -1e-3 % allow for a little imprecision
  fprintf(1, '******likelihood decreased from %6.4f to %6.4f!/n', previous_loglik, loglik);
  decrease = 1;
end

% The following stopping criterion is from Numerical Recipes in C p423
delta_loglik = abs(loglik - previous_loglik);
avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2;
if (delta_loglik / avg_loglik) < threshold, converged = 1; end
end

%% dhmm_logprob
function [loglik, errors] = dhmm_logprob(data, prior, transmat, obsmat)
% LOG_LIK_DHMM Compute the log-likelihood of a dataset using a discrete HMM
% [loglik, errors] = log_lik_dhmm(data, prior, transmat, obsmat)
%
% data{m} or data(m,:) is the m'th sequence
% errors  is a list of the cases which received a loglik of -infinity

    if ~iscell(data)
      data = num2cell(data, 2);
    end
    ncases = length(data);

    loglik = 0;
    errors = [];
    for m=1:ncases
      obslik = multinomial_prob(data{m}, obsmat);
      [alpha, beta, gamma, ll] = fwdback(prior, transmat, obslik, 'fwd_only', 1);
      if ll==-inf
        errors = [errors m];
      end
      loglik = loglik + ll;
    end
end
%% fwdback
function [alpha, beta, gamma, loglik, xi_summed, gamma2] = fwdback(init_state_distrib, ...
   transmat, obslik, varargin)
% FWDBACK Compute the posterior probs. in an HMM using the forwards backwards algo.
%
% [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(init_state_distrib, transmat, obslik, ...)
%
% Notation:
% Y(t) = observation, Q(t) = hidden state, M(t) = mixture variable (for MOG outputs)
% A(t) = discrete input (action) (for POMDP models)
%
% INPUT:
% init_state_distrib(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
%  or transmat{a}(i,j) = Pr(Q(t) = j | Q(t-1)=i, A(t-1)=a) if there are discrete inputs
% obslik(i,t) = Pr(Y(t)| Q(t)=i)
%   (Compute obslik using eval_pdf_xxx on your data sequence first.)
%
% Optional parameters may be passed as 'param_name', param_value pairs.
% Parameter names are shown below; default values in [] - if none, argument is mandatory.
%
% For HMMs with MOG outputs: if you want to compute gamma2, you must specify
% 'obslik2' - obslik(i,j,t) = Pr(Y(t)| Q(t)=i,M(t)=j)  []
% 'mixmat' - mixmat(i,j) = Pr(M(t) = j | Q(t)=i)  []
%  or mixmat{t}(m,q) if not stationary
%
% For HMMs with discrete inputs:
% 'act' - act(t) = action performed at step t
%
% Optional arguments:
% 'fwd_only' - if 1, only do a forwards pass and set beta=[], gamma2=[]  [0]
% 'scaled' - if 1,  normalize alphas and betas to prevent underflow [1]
% 'maximize' - if 1, use max-product instead of sum-product [0]
%
% OUTPUTS:
% alpha(i,t) = p(Q(t)=i | y(1:t)) (or p(Q(t)=i, y(1:t)) if scaled=0)
% beta(i,t) = p(y(t+1:T) | Q(t)=i)*p(y(t+1:T)|y(1:t)) (or p(y(t+1:T) | Q(t)=i) if scaled=0)
% gamma(i,t) = p(Q(t)=i | y(1:T))
% loglik = log p(y(1:T))
% xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:T))  - NO LONGER COMPUTED
% xi_summed(i,j) = sum_{t=}^{T-1} xi(i,j,t)  - changed made by Herbert Jaeger
% gamma2(j,k,t) = p(Q(t)=j, M(t)=k | y(1:T)) (only for MOG  outputs)
%
% If fwd_only = 1, these become
% alpha(i,t) = p(Q(t)=i | y(1:t))
% beta = []
% gamma(i,t) = p(Q(t)=i | y(1:t))
% xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:t))
% gamma2 = []
%
% Note: we only compute xi if it is requested as a return argument, since it can be very large.
% Similarly, we only compute gamma2 on request (and if using MOG outputs).
%
% Examples:
%
% [alpha, beta, gamma, loglik] = fwdback(pi, A, multinomial_prob(sequence, B));
%
% [B, B2] = mixgauss_prob(data, mu, Sigma, mixmat);
% [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(pi, A, B, 'obslik2', B2, 'mixmat', mixmat);

    if 0 % nargout >= 5
      warning('this now returns sum_t xi(i,j,t) not xi(i,j,t)')
    end

    if nargout >= 5, compute_xi = 1; else compute_xi = 0; end
    if nargout >= 6, compute_gamma2 = 1; else compute_gamma2 = 0; end

    [obslik2, mixmat, fwd_only, scaled, act, maximize, compute_xi, compute_gamma2] = ...
       process_options(varargin, ...
           'obslik2', [], 'mixmat', [], ...
           'fwd_only', 0, 'scaled', 1, 'act', [], 'maximize', 0, ...
                       'compute_xi', compute_xi, 'compute_gamma2', compute_gamma2);

    [Q T] = size(obslik);

    if isempty(obslik2)
     compute_gamma2 = 0;
    end

    if isempty(act)
     act = ones(1,T);
     transmat = { transmat } ;
    end

    scale = ones(1,T);

    % scale(t) = Pr(O(t) | O(1:t-1)) = 1/c(t) as defined by Rabiner (1989).
    % Hence prod_t scale(t) = Pr(O(1)) Pr(O(2)|O(1)) Pr(O(3) | O(1:2)) ... = Pr(O(1), ... ,O(T))
    % or log P = sum_t log scale(t).
    % Rabiner suggests multiplying beta(t) by scale(t), but we can instead
    % normalise beta(t) - the constants will cancel when we compute gamma.

    loglik = 0;

    alpha = zeros(Q,T);
    gamma = zeros(Q,T);
    if compute_xi
     xi_summed = zeros(Q,Q);
    else
     xi_summed = [];
    end

    %%%%%%%%% Forwards %%%%%%%%%%

    t = 1;
    alpha(:,1) = init_state_distrib(:) .* obslik(:,t);
    if scaled
     %[alpha(:,t), scale(t)] = normaliseC(alpha(:,t));
     [alpha(:,t), scale(t)] = normalise(alpha(:,t));
    end
    %assert(approxeq(sum(alpha(:,t)),1))
    for t=2:T
     %trans = transmat(:,:,act(t-1))';
     trans = transmat{act(t-1)};
     if maximize
       m = max_mult(trans', alpha(:,t-1));
       %A = repmat(alpha(:,t-1), [1 Q]);
       %m = max(trans .* A, [], 1);
     else
       m = trans' * alpha(:,t-1);
     end
     alpha(:,t) = m(:) .* obslik(:,t);
     if scaled
       %[alpha(:,t), scale(t)] = normaliseC(alpha(:,t));
       [alpha(:,t), scale(t)] = normalise(alpha(:,t));
     end
     if compute_xi & fwd_only  % useful for online EM
       %xi(:,:,t-1) = normaliseC((alpha(:,t-1) * obslik(:,t)') .* trans);
       xi_summed = xi_summed + normalise((alpha(:,t-1) * obslik(:,t)') .* trans);
     end
     %assert(approxeq(sum(alpha(:,t)),1))
    end
    if scaled
     if any(scale==0)
       loglik = -inf;
     else
       loglik = sum(log(scale));
     end
    else
     loglik = log(sum(alpha(:,T)));
    end

    if fwd_only
     gamma = alpha;
     beta = [];
     gamma2 = [];
     return;
    end

    %%%%%%%%% Backwards %%%%%%%%%%

    beta = zeros(Q,T);
    if compute_gamma2
      if iscell(mixmat)
        M = size(mixmat{1},2);
      else
        M = size(mixmat, 2);
      end
     gamma2 = zeros(Q,M,T);
    else
     gamma2 = [];
    end

    beta(:,T) = ones(Q,1);
    %gamma(:,T) = normaliseC(alpha(:,T) .* beta(:,T));
    gamma(:,T) = normalise(alpha(:,T) .* beta(:,T));
    t=T;
    if compute_gamma2
     denom = obslik(:,t) + (obslik(:,t)==0); % replace 0s with 1s before dividing
     if iscell(mixmat)
       gamma2(:,:,t) = obslik2(:,:,t) .* mixmat{t} .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M]);
     else
       gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M]);
     end
     %gamma2(:,:,t) = normaliseC(obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M])); % wrong!
    end
    for t=T-1:-1:1
     b = beta(:,t+1) .* obslik(:,t+1);
     %trans = transmat(:,:,act(t));
     trans = transmat{act(t)};
     if maximize
       B = repmat(b(:)', Q, 1);
       beta(:,t) = max(trans .* B, [], 2);
     else
       beta(:,t) = trans * b;
     end
     if scaled
       %beta(:,t) = normaliseC(beta(:,t));
       beta(:,t) = normalise(beta(:,t));
     end
     %gamma(:,t) = normaliseC(alpha(:,t) .* beta(:,t));
     gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));
     if compute_xi
       %xi(:,:,t) = normaliseC((trans .* (alpha(:,t) * b')));
       xi_summed = xi_summed + normalise((trans .* (alpha(:,t) * b')));
     end
     if compute_gamma2
       denom = obslik(:,t) + (obslik(:,t)==0); % replace 0s with 1s before dividing
       if iscell(mixmat)
         gamma2(:,:,t) = obslik2(:,:,t) .* mixmat{t} .* repmat(gamma(:,t), [1 M]) ./ repmat(denom,  [1 M]);
       else
         gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom,  [1 M]);
       end
       %gamma2(:,:,t) = normaliseC(obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]));
     end
    end

    % We now explain the equation for gamma2
    % Let zt=y(1:t-1,t+1:T) be all observations except y(t)
    % gamma2(Q,M,t) = P(Qt,Mt|yt,zt) = P(yt|Qt,Mt,zt) P(Qt,Mt|zt) / P(yt|zt)
    %                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|zt) / P(yt|zt)
    % Now gamma(Q,t) = P(Qt|yt,zt) = P(yt|Qt) P(Qt|zt) / P(yt|zt)
    % hence
    % P(Qt,Mt|yt,zt) = P(yt|Qt,Mt) P(Mt|Qt) [P(Qt|yt,zt) P(yt|zt) / P(yt|Qt)] / P(yt|zt)
    %                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|yt,zt) / P(yt|Qt)
end

%% normalise
function [M, c] = normalise(M)
% NORMALISE Make the entries of a (multidimensional) array sum to 1
% [M, c] = normalise(M)

c = sum(M(:));
% Set any zeros to one before dividing
d = c + (c==0);
M = M / d;

%if c==0
%? tiny = exp(-700);
%? M = M / (c+tiny);
%else
% M = M / (c);
%end
end

%% process options
% PROCESS_OPTIONS - Processes options passed to a Matlab function.
%                   This function provides a simple means of
%                   parsing attribute-value options.  Each option is
%                   named by a unique string and is given a default
%                   value.
%
% Usage:  [var1, var2, ..., varn[, unused]] = ...
%           process_options(args, ...
%                           str1, def1, str2, def2, ..., strn, defn)
%
% Arguments:   
%            args            - a cell array of input arguments, such
%                              as that provided by VARARGIN.  Its contents
%                              should alternate between strings and
%                              values.
%            str1, ..., strn - Strings that are associated with a 
%                              particular variable
%            def1, ..., defn - Default values returned if no option
%                              is supplied
%
% Returns:
%            var1, ..., varn - values to be assigned to variables
%            unused          - an optional cell array of those 
%                              string-value pairs that were unused;
%                              if this is not supplied, then a
%                              warning will be issued for each
%                              option in args that lacked a match.
%
% Examples:
%
% Suppose we wish to define a Matlab function 'func' that has
% required parameters x and y, and optional arguments 'u' and 'v'.
% With the definition
%
%   function y = func(x, y, varargin)
%
%     [u, v] = process_options(varargin, 'u', 0, 'v', 1);
%
% calling func(0, 1, 'v', 2) will assign 0 to x, 1 to y, 0 to u, and 2
% to v.  The parameter names are insensitive to case; calling 
% func(0, 1, 'V', 2) has the same effect.  The function call
% 
%   func(0, 1, 'u', 5, 'z', 2);
%
% will result in u having the value 5 and v having value 1, but
% will issue a warning that the 'z' option has not been used.  On
% the other hand, if func is defined as
%
%   function y = func(x, y, varargin)
%
%     [u, v, unused_args] = process_options(varargin, 'u', 0, 'v', 1);
%
% then the call func(0, 1, 'u', 5, 'z', 2) will yield no warning,
% and unused_args will have the value {'z', 2}.  This behaviour is
% useful for functions with options that invoke other functions
% with options; all options can be passed to the outer function and
% its unprocessed arguments can be passed to the inner function.

% Copyright (C) 2002 Mark A. Paskin
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
% USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varargout] = process_options(args, varargin)

% Check the number of input arguments
n = length(varargin);
if (mod(n, 2))
  error('Each option must be a string/value pair.');
end

% Check the number of supplied output arguments
if (nargout < (n / 2))
  error('Insufficient number of output arguments given');
elseif (nargout == (n / 2))
  warn = 1;
  nout = n / 2;
else
  warn = 0;
  nout = n / 2 + 1;
end

% Set outputs to be defaults
varargout = cell(1, nout);
for i=2:2:n
  varargout{i/2} = varargin{i};
end

% Now process all arguments
nunused = 0;
for i=1:2:length(args)
  found = 0;
  for j=1:2:n
    if strcmpi(args{i}, varargin{j})
      varargout{(j + 1)/2} = args{i + 1};
      found = 1;
      break;
    end
  end
  if (~found)
    if (warn)
      warning(sprintf('Option ''%s'' not used.', args{i}));
      args{i}
    else
      nunused = nunused + 1;
      unused{2 * nunused - 1} = args{i};
      unused{2 * nunused} = args{i + 1};
    end
  end
end

% Assign the unused arguments
if (~warn)
  if (nunused)
    varargout{nout} = unused;
  else
    varargout{nout} = cell(0);
  end
end
end

%% multinomial_prob
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 已知：观察序列data和观察值概率矩阵obsmat
% 求解：条件概率矩阵B
function B = multinomial_prob(data, obsmat)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EVAL_PDF_COND_MULTINOMIAL Evaluate pdf of conditional multinomial 
    % function B = eval_pdf_cond_multinomial(data, obsmat)
    % From the MIT-toolbox by Kevin Murphy, 2003.
    % Notation: Y = observation (O values), Q = conditioning variable (K values)
    %
    % Inputs:
    % data(t) = t'th observation - must be an integer in {1,2,...,K}: cannot be 0!
    % obsmat(i,o) = Pr(Y(t)=o | Q(t)=i)
    %
    % Output:
    % B(i,t) = Pr(y(t) | Q(t)=i)

    [Q O] = size(obsmat);
    T = prod(size(data)); % length(data);
    B = zeros(Q,T);

    for t=1:T
      B(:,t) = obsmat(:, data(t));
    end
end