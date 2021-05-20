function [alpha,beta]=BrehensModel(observations,reward,K,r,dr,v,dv)

%% define transition models and emission models
p_v_transition=@(vi,v_i1) normpdf(v_i1,vi,K)*dv/sum(normpdf(v,vi,K)*dv);
p_r_emission=@(y,r_i1) binopdf(y,1,r_i1)/sum(binopdf(y,1,r));
% p_r_transion is at the bottom of the script

%% initialize
trials_to_try=length(observations);

alpha=zeros(number,number,length(trials_to_try));
alpha(:,:,1)=1;
alpha(:,:,1)=alpha(:,:,1)./sum(sum(alpha(:,:,1)));

beta=zeros(number,number,length(trials_to_try)); %one step transition for r

beta_r_tmp=zeros(number,number,number,length(trials_to_try))+0.5;
beta_v=zeros(number,number,length(trials_to_try)); %one step transition for v
beta_v_tmp=zeros(number,number,number,length(trials_to_try));

%% iteratively predict, observe, and update
for t=1:trials_to_try
   disp(t)  
   alpha(:,:,t)= alpha(:,:,max([t-1,1]));
   
   if ~observations(t)
       % copy from previous
       beta(:,:,t)=beta(:,:,max([t-1,1]));
       beta_v(:,:,t)=beta_v(:,:,max([t-1,1]));
       beta_r_tmp(:,:,:,t)=beta_r_tmp(:,:,:,max([t-1,1]));
       beta_v_tmp(:,:,:,t)=beta_v_tmp(:,:,:,max([t-1,1]));
       continue
   end
   
   % get one step transition prediction for v
   for v_index=1:number
       for v_i1_index=1:number
           vi=v(v_index);
           v_i1=v(v_i1_index);
           beta_v_tmp(:,v_index,v_i1_index,t)=alpha(:,v_index,t)*p_v_transition(vi,v_i1);
       end
   end
   beta_v(:,:,t)=squeeze(sum(beta_v_tmp(:,:,:,t),2));
   
   % get one step transition prediction for r
   for v_index=1:number
       v_i1=v(v_index);
       for r_index=1:number
       for r_i1_index=1:number
           ri=r(r_index);
           r_i1=r(r_i1_index);
           [p_r,a,b]=p_r_transition(r_i1,ri,exp(v_i1),dr,r);
           beta_r_tmp(r_index,r_i1_index,v_index,t)=beta_v(r_index,v_index,t)*p_r;
       end
       end
   end
   beta(:,:,t)=squeeze(sum(beta_r_tmp(:,:,:,t),1));
   % normalize beta to sum to 1
   beta(:,:,t)=beta(:,:,t)/sum(sum(beta(:,:,t)));

   % update by one step emission model for r
   for r_index=1:number
       r_i1=r(r_index);
       alpha(r_index,:,t)=beta(r_index,:,t)*p_r_emission(reward,r_i1);
   end
   
   % normalize alpha to sum to 1
   alpha(:,:,t)=alpha(:,:,t)/sum(sum(alpha(:,:,t)));
end

for t=1:trials_to_try
    if ~observations(t)
        alpha(:,:,t)=nan;
        beta(:,:,t)=nan;
    end
end
end

function [p,a,b]=p_r_transition(x,mean,width,dr,r)
    p=0;
    a=1/width*mean;
    b=1/width-a;
    if and(a>0,b>0)
        p=betapdf(x,a,b)*dr/sum(betapdf(r,a,b)*dr);
    end
    if isinf(p)
        p=0;
    end
end