%% load data

filepath='/home/shijiegu/Documents/SpatialBanditTask';
csv_file='senor_clean_contingencies_only_parsed_depletion_data.csv';
T = readtable(fullfile(filepath,csv_file));
session=T.session;
contingency=T.contingency;
trials_to_try=900;

%% Tim Brehens 2007
number=50;
r=linspace(0.01,0.99,number); % avoid 0 and 1 as these can give Inf in beta distribution
v=linspace(-4,log(1/2),number);
k=exp(linspace(-4,1,number));

obs={};
obs{1}=[or(T.leaf==1,T.leaf==2),T.reward];
obs{2}=[or(T.leaf==3,T.leaf==4),T.reward];
obs{3}=[or(T.leaf==5,T.leaf==6),T.reward];
for l=1:6
    obs{l+3}=[T.leaf==l,T.reward];
end
for o_index=1:length(obs)
    obs{o_index}=obs{o_index}(1:(1+trials_to_try),:);
end
alphas={};
betas={};
%%
version=1;
parfor o_index=1:length(obs)
    observations=obs{o_index}(:,1);
    reward=obs{o_index}(:,2);
    [alpha,beta]=BrehensModel(observations,reward,k,r,v,version);
    alphas{o_index}=alpha;
    betas{o_index}=beta;
end
save(fullfile('/home/shijiegu/Documents/SpatialBanditTask','trial1_to_900_version1_depletion.mat'),'alphas','betas','-v7.3')
%%
colors=[0, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
    0.4660, 0.6740, 0.1880;
    0.3010, 0.7450, 0.9330];
%%
cont_ind={};
cont_ind{1}{1}=[1,2]; cont_ind{1}{2}=[3,4];
cont_ind{2}{1}=[5,6]; cont_ind{2}{2}=[7,8];
cont_ind{3}{1}=[9,10]; cont_ind{3}{2}=[11,12];

cont_ind{4}{1}=[1,2]; cont_ind{4}{2}=cont_ind{4}{1};
cont_ind{5}{1}=[3,4]; cont_ind{5}{2}=cont_ind{5}{1};
cont_ind{6}{1}=[5,6]; cont_ind{6}{2}=cont_ind{6}{1};

cont_str=num2str(T.contingency(1:trials_to_try));
cont={};
for cont_i=1:6
    cont{cont_i}=zeros(trials_to_try,3);
    prob1=str2num(cont_str(:,cont_ind{cont_i}{1}));
    prob2=str2num(cont_str(:,cont_ind{cont_i}{2}));
    cont{cont_i}(:,2)=min([prob1,prob2],[],2)/100;
    cont{cont_i}(:,3)=max([prob1,prob2],[],2)/100;
    cont{cont_i}(:,1)=1/2*(cont{cont_i}(:,2)+cont{cont_i}(:,3));
end
%%
f=figure;
f.Position = [10 10 1000 600];
axe_cells=cell(1,2);
titles={'stem1','stem 2','stem 3'};
axe_cells{1}=subplot(2,1,1);

% plot
for o_index=1:3
    observations=obs{o_index};
    alpha=alphas{o_index};
    beta=betas{o_index};
    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    
    
    r_hat=sum(r*squeeze(sum(sum(alpha(:,:,:,1:trials_to_try),3),2)),1);
    r_pred=sum(r*squeeze(sum(sum(beta(:,:,:,1:trials_to_try),3),2)),1);
    lr=(diff([0,r_hat]))./(observations(1:trials_to_try,2)'-r_hat(1:end));
    lr=(r_pred(2:end)-r_hat(2:end));
    plot(r_pred,'color',c1)
    hold on

    axis tight
    
    ylim([0,1])
end
title('inferred reward rate')

axe_cells{2}=subplot(2,1,2);
for o_index=1:3
    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    fill([1:trials_to_try fliplr(1:trials_to_try)],...
        [cont{o_index}(:,2)' fliplr(cont{o_index}(:,3)')],c1,'EdgeColor','None','FaceAlpha',0.5)
    hold on
end
for o_index=1:3
    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    plot(cont{o_index}(:,1),':','color',c2,'LineWidth',3)
    hold on
end
axis tight
ylim([0,1])
title('ground truth contingency')

for axe_index=1:2
    axes(axe_cells{axe_index});
    % plot contingency boundaries
    boundary=find(diff([0;contingency]));
    for b=1:length(boundary)
        if boundary(b)<max(find(1:trials_to_try))
            plot([boundary(b),boundary(b)],[0,1],'color',[0.1,0.1,0.1])
            %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
            %set(h,'Rotation',90);
        end
    end
    % plot session boundaries
    boundary=find(diff([0;session]));
    for b=1:length(boundary)
        if boundary(b)<max(find(1:trials_to_try))
            plot([boundary(b),boundary(b)],[0,1],'color',[0.8,0.1,0.1])
            %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
            %set(h,'Rotation',90);
        end
    end
end
legend(axe_cells{1},'stem 1','stem 2','stem 3')
hleg = legend('show');
legend_to_plot=hleg.String(1:3);
hleg.String = legend_to_plot;

legend(axe_cells{2},'stem 1','stem 2','stem 3','stem 1 mean','stem 2 mean','stem 3 mean')
hleg = legend('show');
legend_to_plot=hleg.String(1:6);
hleg.String = legend_to_plot;
