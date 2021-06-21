number=20;
r=linspace(0.01,0.99,number); % avoid 0 and 1 as these can give Inf in beta distribution
v=linspace(-4,log(1/2),number);
k=exp(linspace(-4,1,number));
y=[0,1];

number=length(r);
dr=r(2)-r(1);
dv=v(2)-v(1);
%% define transition models and emission models
p_v_transition=@(vi,v_i1,ki) normpdf(v_i1,vi,ki)*dv/sum(normpdf(v,vi,ki)*dv);
p_r_emission=@(y,r_i1) binopdf(y,1,r_i1);%/sum(binopdf(y,1,r));
% p_r_transion is at the bottom of the script

%% make ground truth data
k_gt=1;
v_gt=zeros(100,1);
v_gt(1)=-1;
r_gt=zeros(100,1);
r_gt(1)=-1;
y_gt=zeros(100,1);
y_gt(1)=1;
for t=2:100
    rng(t*3)
    rand_num=rand;
    v_pdf=cumsum([p_v_transition(v_gt(t-1),v,k_gt)]);
    v_gt(t)=v(find(v_pdf>=rand_num,1));
    
    rng(t*3+1)
    rand_num=rand;
    [tmp,a,b]=p_r_transition(r,r_gt(t-1),exp(v_gt(t)),dr,r);
    if tmp==0
        r_gt(t)=r(1);
    else
        r_pdf=cumsum([tmp]);
        r_gt(t)=r(find(r_pdf>=rand_num,1));
    end
    
    rng(t*3)
    rand_num=rand;
    y_pdf=cumsum([p_r_emission(y(1),r_gt(t));p_r_emission(y(2),r_gt(t))]);    
    y_gt(t)=y(find(y_pdf>=rand_num,1));
    
end

%% run the model
version=1;
[alpha,beta]=BrehensModel(zeros(1,100)+1,y_gt,k,r,v,version);
%% plot result
r_hat=sum(r*squeeze(sum(sum(alpha(:,:,:,1:100),3),2)),1);
v_hat=sum(v*squeeze(sum(sum(alpha(:,:,:,1:100),3),1)),1);
k_hat=sum(k*squeeze(sum(sum(alpha(:,:,:,1:100),2),1)),1);
figure;
plot(r_hat(2:end),'color',[0.8,0.05,0.05]);
hold on;
plot(r_gt(2:end),'color',[1,0.5,0.5]);
plot(exp(v_hat(2:end)),'color',[0.05,0.05,0.8]);
plot(exp(v_gt(2:end)),'color',[0.5,0.5,1])
plot(k_hat(2:end)','color',[0.05,0.8,0.05]);
plot(1:length(k_hat(2:end)),zeros(1,length(k_hat(2:end)))+k_gt,'color',[0.5,1,0.5])
legend('mean(r), est','r gt','mean(V), est','V gt','mean(K), est','K gt')
xlabel('trial')
ylabel('a.u.')