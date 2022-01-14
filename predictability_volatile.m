%% load data
filepath='/Users/shijiegu/Documents/Second-Year/w_Alison';
csv_file='senor_clean_contingencies_only_parsed_data.csv';
T = readtable(fullfile(filepath,csv_file));
%%
trial_ind=T.date==20201028;
C=T.contingency;
C=C(trial_ind);
C_set=unique(C);
common_num=zeros(1,length(C_set));
pattern_switch_back=zeros(1,62);
pattern_switch_back(1)=1;
pattern_switch_back(end)=1;

for c=1%:length(C_set)
    trial_ind1=double(C(trial_ind)==C_set(c));
    k = findpattern(trial_ind1,pattern_switch_back);
    if k>0; common_num(c)=k; end
end

%% fit Piray 2020

trial_ind=T.date==20201028;
o = T.reward(trial_ind);
lambda_v = .1;
lambda_u = .1;

contingency=T.contingency;
pmean=0.01*(80+50)/2;
v0 = 1/mean(diff(find(diff(contingency))));
u0 = pmean*(1-pmean);
np = 100;

config = struct('lambda_v',lambda_v,'lambda_u',lambda_u,'v0',v0,'u0',u0,'nparticles',np);
[vol,unp,lr,val,unc] = model_pf(o,config);
%%
choice=T.leaf(trial_ind);
reward=T.reward(trial_ind);
%%
figure;
% plot volatility and unpredictability
yyaxis left
plot(vol)
hold on
plot(unp)
%plot(val)
legend('vol','unp')

% plot reward
yyaxis right
plot(choice,'*','color',[0.2,0.2,0.2])
rewarded_choice=choice.*reward;
plot(find(rewarded_choice),rewarded_choice(rewarded_choice>0),'*','color','r')

% plot boundary
boundary=find(diff([0;contingency]));
for b=1:length(boundary)
    if boundary(b)<max(find(trial_ind))
        plot([boundary(b),boundary(b)],[0,6])
        text(boundary(b),4.5,num2str(T.contingency(boundary(b))))
    end
end



hleg = legend('show');
legend_to_plot=hleg.String(1:3);
hleg.String = legend_to_plot;

axis tight
