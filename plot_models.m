%%
model_type='q';
filepath='/home/shijiegu/Documents/SpatialBanditTask';
filename='senor_q_learning_version1_session_depletion.mat';
load(fullfile(filepath,filename))
csv_file='senor_clean_contingencies_only_parsed_depletion_data.csv';
T = readtable(fullfile(filepath,csv_file));
%%
trials_to_try=900;
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
colors=[0, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
    0.4660, 0.6740, 0.1880;
    0.3010, 0.7450, 0.9330];

number=50;
r=linspace(0.01,0.99,number); % avoid 0 and 1 as these can give Inf in beta distribution
v=linspace(-4,log(1/2),number);
k=exp(linspace(-4,1,number));

%%
cont_ind={};
cont_ind{1}{1}=[1,2]; cont_ind{1}{2}=[3,4];
cont_ind{2}{1}=[5,6]; cont_ind{2}{2}=[7,8];
cont_ind{3}{1}=[9,10]; cont_ind{3}{2}=[11,12];

cont_ind{4}{1}=[1,2]; cont_ind{4}{2}=cont_ind{4}{1};
cont_ind{5}{1}=[3,4]; cont_ind{5}{2}=cont_ind{5}{1};
cont_ind{6}{1}=[5,6]; cont_ind{6}{2}=cont_ind{6}{1};

cont_ind{7}{1}=[7,8]; cont_ind{7}{2}=cont_ind{7}{1};
cont_ind{8}{1}=[9,10]; cont_ind{8}{2}=cont_ind{8}{1};
cont_ind{9}{1}=[11,12]; cont_ind{9}{2}=cont_ind{9}{1};

contingency=T.contingency;
session=T.session;
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
%% calculate likelihood for stem
Q_all=zeros(9,trials_to_try);
if strcmp(model_type,'q')
    t_count=0;
    for s_index=1:numel(Q)
        for t=1:size(Q{s_index},3)
            Q_all(4:9,t_count+t)=(reshape(Q{s_index}(:,:,t)',6,1)+1)/2; 
            %scale it so that it lies between 0 and 1
        end
        t_count=t_count+size(Q{s_index},3);
    end
    Q_all(1,:)=mean(Q_all(4:5,:));
    Q_all(2,:)=mean(Q_all(6:7,:));
    Q_all(3,:)=mean(Q_all(8:9,:));
    r_hat_all=Q_all;
else
    Q=zeros(9,trials_to_try);
    for o_index=1:9
        alpha=alphas{o_index};
        r_hat=sum(r*squeeze(sum(sum(alpha(:,:,:,1:trials_to_try),3),2)),1);
        r_hat_tmp=fillmissing(r_hat,'previous');
        r_hat_tmp=fillmissing(r_hat_tmp,'next');
        Q(o_index,:)=r_hat_tmp;
    end
    stem_choice=floor(T.leaf/2)+mod(T.leaf,2);
    leaf_choice=T.leaf;
    
    boundary=find(diff([0;session]));
    lik_stem=[];
    lik_leaf=[];
    for b=1:length(boundary)
        if boundary(b)<max(find(1:trials_to_try))
            session_begin=boundary(b);
            session_end=min([boundary(b+1),trials_to_try]);
            
            lik_stem=[lik_stem,loglikelihood(Q(1:3,session_begin:session_end),stem_choice(session_begin:session_end))];
            
            lik_leaf=[lik_leaf,loglikelihood(Q(4:9,session_begin:session_end),leaf_choice(session_begin:session_end))];
        end
    end
end
%% calculate r_hat
if ~strcmp(model_type,'q')
    r_hat_all=zeros(9,trials_to_try);
    lr_all=zeros(9,trials_to_try);
    for o_index=1:9
        observations=obs{o_index};
        alpha=alphas{o_index};
        r_hat=sum(r*squeeze(sum(sum(alpha(:,:,:,1:trials_to_try),3),2)),1);
        r_hat_all(o_index,:)=r_hat;
        notnan_index=~isnan(r_hat);
        lr_all(o_index,find(notnan_index))=(diff([0,r_hat(notnan_index)]))./(observations(notnan_index,2)'-r_hat(notnan_index));
    end
    save(fullfile(filepath,filename),'lr_all','r_hat_all','lik_stem','lik_leaf','-append');
else
    lr=
end
%%
f=figure;
f.Position = [10 10 1000 600];
axe_cells=cell(1,2);
titles={'stem1','stem 2','stem 3'};
axe_cells{1}=subplot(3,1,1);

% plot
for o_index=1:3
    observations=obs{o_index};

    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    r_hat=r_hat_all(o_index,:);
     
    plot(mean([fillmissing(r_hat_all((o_index+1)*2,:),'previous');fillmissing(r_hat_all((o_index+1)*2+1,:),'previous')],1),':','color',c1)
    hold on
    rewarded=double(observations(1:trials_to_try,2)>0);
    rewarded(rewarded==0)=nan;
    unrewarded=double(observations(1:trials_to_try,2)<=0);
    unrewarded(unrewarded==0)=nan;
    
    plot(r_hat.*rewarded','.','color',c1,'MarkerSize',5)
    hold on
    plot(r_hat.*unrewarded','o','color',c1,'MarkerSize',2)
end
axis tight
ylim([0,1])
title('inferred reward rate')

axe_cells{2}=subplot(3,1,2);
for o_index=1:3
    c1=colors(o_index,:);
    observations=obs{o_index};
    
    r_hat=r_hat_all(o_index,:);
    notnan_index=~isnan(r_hat);
    lr=lr_all(o_index,:);
    plot(find(notnan_index),lr(notnan_index),'color',c1)
    hold on
end
axis tight
ylim([0,1])
title('inferred learning rate')

axe_cells{3}=subplot(3,1,3);
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

for axe_index=2
    axes(axe_cells{axe_index});
    % plot contingency boundaries
%     boundary=find(diff([0;contingency]));
%     for b=1:length(boundary)
%         if boundary(b)<max(find(1:trials_to_try))
%             plot([boundary(b),boundary(b)],[0,1],'color',[0.1,0.1,0.1])
%             hold on
%             %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
%             %set(h,'Rotation',90);
%         end
%     end
    % plot session boundaries
    boundary=find(diff([0;session]));
    for b=1:length(boundary)
        if boundary(b)<max(find(1:trials_to_try))
            plot([boundary(b),boundary(b)],[0,1],'color',[0.8,0.1,0.1])
            h=text(boundary(b),0,num2str(lik_stem(b)));
            %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
            set(h,'Rotation',90);
        end
    end
end

legend(axe_cells{1},'leaf mean','stem 1','','leaf mean','stem 2','','leaf mean','stem 3','')
hleg = legend('show');
legend_to_plot=hleg.String(1:6);
hleg.String = legend_to_plot;

legend(axe_cells{2},'stem 1','stem 2','stem 3')
hleg = legend('show');
legend_to_plot=hleg.String(1:3);
hleg.String = legend_to_plot;

legend(axe_cells{3},'stem 1','stem 2','stem 3','stem 1 mean','stem 2 mean','stem 3 mean')
hleg = legend('show');
legend_to_plot=hleg.String(1:3);
hleg.String = legend_to_plot;

h=suptitle(csv_file);
h.Interpreter = 'none';

%%
f=figure;
f.Position = [10 10 1000 600];
axe_cells=cell(1,2);
titles={'leaf1','leaf2','leaf3','leaf4','leaf5','leaf6'};
axe_cells{1}=subplot(3,1,1);

% plot
for o_index=4:9
    observations=obs{o_index};
    c1=colors(o_index-3,:);
    c2=min(colors(o_index-3,:).*[2,2,2],[1,1,1]);
    
    r_hat=r_hat_all(o_index,:);
    %r_hat=fillmissing(r_hat,'previous');
    %plot(r_hat,'.','color',c1)
    %hold on
    
    rewarded=double(observations(1:trials_to_try,2)>0);
    rewarded(rewarded==0)=nan;
    unrewarded=double(observations(1:trials_to_try,2)<=0);
    unrewarded(unrewarded==0)=nan;
    
    plot(r_hat.*rewarded','.','color',c1,'MarkerSize',5)
    hold on
    plot(r_hat.*unrewarded','o','color',c1,'MarkerSize',2)
end
axis tight
ylim([0,1])

title('inferred reward rate')

axe_cells{2}=subplot(3,1,2);
for o_index=4:9
    c1=colors(o_index-3,:);
    observations=obs{o_index};
    
    r_hat=r_hat_all(o_index,:);
    notnan_index=~isnan(r_hat);
    lr=lr_all(o_index,:);
    plot(find(notnan_index),lr(notnan_index),'color',c1)
    hold on
end
axis tight
ylim([0,1])
title('inferred learning rate')

axe_cells{3}=subplot(3,1,3);
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

for axe_index=2
    axes(axe_cells{axe_index});
%   not applicable to depletion contingency
%     % plot contingency boundaries
%     boundary=find(diff([0;contingency]));
%     for b=1:length(boundary)
%         if boundary(b)<max(find(1:trials_to_try))
%             plot([boundary(b),boundary(b)],[0,1],'color',[0.1,0.1,0.1])
%             %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
%             %set(h,'Rotation',90);
%         end
%     end
    % plot session boundaries
    boundary=find(diff([0;session]));
    for b=1:length(boundary)
        if boundary(b)<max(find(1:trials_to_try))
            plot([boundary(b),boundary(b)],[0,1],'color',[0.8,0.1,0.1])
            h=text(boundary(b),0,num2str(lik_leaf(b)));
            %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
            set(h,'Rotation',90);
        end
    end
end

legend(axe_cells{1},'leaf 1','','leaf 2','','leaf 3','','leaf 4','','leaf 5','','leaf 6','')
hleg = legend('show');
%legend_to_plot=hleg.String(1:end);
%hleg.String = legend_to_plot;

legend(axe_cells{2},'leaf 1','leaf 2','leaf 3','leaf 4','leaf 5','leaf 6')
hleg = legend('show');
legend_to_plot=hleg.String(1:6);
hleg.String = legend_to_plot;

legend(axe_cells{3},'stem 1','stem 2','stem 3','stem 1 mean','stem 2 mean','stem 3 mean')
hleg = legend('show');
legend_to_plot=hleg.String(1:3);
hleg.String = legend_to_plot;

h=suptitle(csv_file);
h.Interpreter = 'none';