function [alphas,betas,lr_all,r_pred_all,r_hat_all,lik,...
    boundary]=runBrehens(filepath_csv,filepath_result,model_version,trials_to_try)
% model version: 1 or 2 or 3 (ideal observer model)

%% load data
T = readtable(filepath_csv);
session=T.session;
contingency=T.contingency;

%% Tim Brehens 2007
number=20;
r=linspace(0,1,number); % avoid 0 and 1 as these can give Inf in beta distribution
r=r+r(2)-r(1);
r=r(1:end-1);
v=log(linspace(exp(-4),1/2,number));
k=linspace(0.01,1/2,10);

% make independent observations for each stem and leaf
obs={};
full_obs={};
obs{1}=[or(T.leaf==1,T.leaf==2),T.reward];
obs{2}=[or(T.leaf==3,T.leaf==4),T.reward];
obs{3}=[or(T.leaf==5,T.leaf==6),T.reward];
for l=1:6
    obs{l+3}=[T.leaf==l,T.reward];
    full_obs{l+3}=obs{ceil(l/2)}(trials_to_try,1);
end
% trim data
for o_index=1:length(obs)
    obs{o_index}=obs{o_index}(trials_to_try,:);
end

alphas={};
betas={};
%% for now, just run leaves
parfor o_index=4:length(obs)
    observations=obs{o_index}(:,1);
    full_observations=full_obs{o_index};
    reward=obs{o_index}(:,2);
    [alpha,beta]=BrehensModel(observations,reward,k,r,v,model_version,full_observations);
    alphas{o_index}=alpha;
    betas{o_index}=beta;
end
% load(filepath_result,'alphas','betas')
%% get r_pred_all, r_hat_all, and calculate likelihood
r_hat_all=zeros(9,length(trials_to_try));
r_pred_all=zeros(9,length(trials_to_try));
lr_all=zeros(9,length(trials_to_try));
for o_index=4:9
    beta=betas{o_index};
    r_pred=sum(r*squeeze(sum(sum(beta(:,:,:,1:length(trials_to_try)),3),2)),1);
    r_pred_all(o_index,:)=r_pred;
    
    alpha=alphas{o_index};
    r_hat=sum(r*squeeze(sum(sum(alpha(:,:,:,1:length(trials_to_try)),3),2)),1);
    r_hat_all(o_index,:)=r_hat;
    
    rewards=obs{o_index}(:,2).*obs{o_index}(:,1);
    lr_all(o_index,:)=abs((diff([0,r_hat]))./(rewards'-r_hat));
end
for o_index=1:3
    r_hat_all(o_index,:)=0.5*(r_hat_all((o_index+1)*2,:)+r_hat_all((o_index+1)*2+1,:));
    lr_all(o_index,:)=0.5*(lr_all((o_index+1)*2,:)+lr_all((o_index+1)*2+1,:));
    r_pred_all(o_index,:)=0.5*(r_pred_all((o_index+1)*2,:)+r_pred_all((o_index+1)*2+1,:));
end

stem_choice=floor(T.leaf/2)+mod(T.leaf,2);
leaf_choice=T.leaf;
    
boundary=find(diff([0;session]));
lik=[];
r_pred_modified_all=[];
params=[];
boundaries=[];
for b=find(boundary>=min(trials_to_try),1):find(boundary<=max(trials_to_try),1,'last')
    session_begin=boundary(b);
    session_end=min([boundary(b+1),max(trials_to_try)]);
    
    stem_choices=stem_choice(session_begin:session_end);
    leaf_choices=leaf_choice(session_begin:session_end);
    Q_leaf=r_pred_all(4:9,(session_begin-trials_to_try(1)+1):(session_end-trials_to_try(1)+1));
    Q_stem=r_pred_all(1:3,(session_begin-trials_to_try(1)+1):(session_end-trials_to_try(1)+1));
    
    [lik,bias,beta_stem,beta_leaf,q_stem_modified,q_leaf_modified]=loglikelihood_beta(Q_stem,Q_leaf,stem_choices,leaf_choices);
    lik=[lik,lik_];
    params=[params;[bias,beta_stem,beta_leaf]];
    boundaries=[boundaries,boundary(b)];
    r_pred_modified=[q_leaf_modified;q_stem_modified];
    r_pred_modified_all=[r_pred_modified_all,r_pred_modified];
end

%% save matfile
if ~isempty(filepath_result)
    save(filepath_result,'alphas','betas','lr_all','r_pred_all','r_hat_all','r_pred_modified_all','lik','params','boundaries','filepath_csv','trials_to_try','r','v','k','-v7.3')
end
