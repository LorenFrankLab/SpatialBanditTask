x=linspace(0,1,50); % reward rate
ws=linspace(exp(-4),1/2,100); % width of beta

figure;
subplot(2,1,1)
for w_ind=1:length(ws)
    w=ws(w_ind);
    
    % if current reward rate rate (r_i) is 0.8, 
    % find (r_i+1) by beta
    [p,a,b]=p_r_transition(x,0.8,w,x(2)-x(1),x);
    plot(x,p);

    hold on
    % plot mean
    plot([sum(p.*x),sum(p.*x)],[0,0.2])
end

subplot(2,1,2)
for w_ind=1:length(ws)
    w=ws(w_ind);
    
    % if current reward rate rate (r_i) is 0.1, 
    % find (r_i+1) by beta
    [p,a,b]=p_r_transition(x,0.1,w,x(2)-x(1),x);
    plot(x,p);

    hold on
    % plot mean
    plot([sum(p.*x),sum(p.*x)],[0,0.2])
end
%% Notes
% it seems like this asymmetry in transitioning can introduce model
% behavior that is not necessarily true to rats' behavior
% when the current reward rate is low, future estimate tend to be low while
% when the current reward rate is high, future estimate tend to be high
% too.
%%
figure;
a=0.5;
b=0.8;
v=1;

p_all=betapdf(linspace(0.001,0.999,100),a,b);
p_all=p_all./sum(p_all);
plot(p_all)
hold on;

for t=1:10
    s=v*mean([a,b])+(1-v)*1
    a_=a/s;
    b_=b/s;
    p_all=betapdf(linspace(0.001,0.999,100),a_,b_);
    p_all=p_all./sum(p_all);
    plot(p_all)
    
    a=a_;
    b=b_;
end

%% The following block shifts the pdf to make it more mean preserving
number=20;
r=linspace(0,1,number); % avoid 0 and 1 as these can give Inf in beta distribution
v=log(linspace(exp(-4),1/2,number));

test=zeros(length(r),length(r));
test_shifted=test;
mean_analytic=[];
mean_numer=[];
mean_shifted=[];
var_analytic=[];
var_numer=[];
var_shifted=[];
for r_ind=1:length(r)
    ri=r(r_ind);
    [test(r_ind,:),a,b]=p_r_transition(r,ri,exp(v(20)),dr,r);

    mean_analytic(r_ind)=a/(a+b);
    var_analytic(r_ind)=a*b/((a+b)^2*(a+b+1));
    
    mean_numer(r_ind)=sum(test(r_ind,:).*r);
    var_numer(r_ind)=sum(test(r_ind,:).*(r-mean_numer(r_ind)).^2);
    
    test_shifted(r_ind,:)=test(r_ind,:);
    if and(r_ind>=2,r_ind<=length(r)-1)
        if mean_analytic(r_ind)<mean_numer(r_ind)
            DELTA=mean_numer(r_ind)-mean_analytic(r_ind);
            tmp=zeros(1,length(r));
            ind_future=min([length(r),(r_ind+1)]):length(r);
            ind_past=1:r_ind-1;
            tmp(ind_future)=1;
            tmp(ind_past)=-sum(length(ind_future))/length(ind_past);
            DELTA_P=DELTA/sum(r.*tmp);
            
            test_shifted(r_ind,ind_future)=test_shifted(r_ind,ind_future)-DELTA_P;
            test_shifted(r_ind,ind_past)=test(r_ind,ind_past)-DELTA_P*(tmp(ind_past));
            
            to_minus=sum(test_shifted(r_ind,test_shifted(r_ind,:)<=0));
            ind_to_minus=find(test_shifted(r_ind,ind_future)>=-to_minus,1,'last');
            test_shifted(r_ind,ind_future(ind_to_minus))=test_shifted(r_ind,ind_future(ind_to_minus))+to_minus;
            test_shifted(r_ind,test_shifted(r_ind,:)<=0)=0;
        else
            DELTA=-mean_numer(r_ind)+mean_analytic(r_ind);
            ind_future=(r_ind+1):length(r);
            ind_past=1:max([1,r_ind-1]);
            tmp=zeros(1,length(r));
            tmp(ind_past)=-1;
            tmp(ind_future)=sum(length(ind_past))/length(ind_future);
            DELTA_P=DELTA/sum(r.*tmp);
           
            test_shifted(r_ind,ind_past)=test_shifted(r_ind,ind_past)-DELTA_P;
            test_shifted(r_ind,ind_future)=test(r_ind,ind_future)+DELTA_P*(tmp(ind_future));
            
            to_minus=sum(test_shifted(r_ind,test_shifted(r_ind,:)<=0));
            ind_to_minus=find(test_shifted(r_ind,ind_past)>=-to_minus,1);
            
            test_shifted(r_ind,ind_past(ind_to_minus))=test_shifted(r_ind,ind_past(ind_to_minus))+to_minus;
            test_shifted(r_ind,test_shifted(r_ind,:)<=0)=0;
        end
    end
         
    mean_shifted(r_ind)=sum(test_shifted(r_ind,:).*r);
    var_shifted(r_ind)=sum(test_shifted(r_ind,:).*(r-mean_shifted(r_ind)).^2);
    
    
    
end
figure;
% plot(var_analytic);
% hold on;
% plot(var_numer);
% plot(var_shifted)
plot(mean_analytic);
hold on;
plot(mean_numer);
plot(mean_shifted)
