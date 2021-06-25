function plot_model(model_type,filepath_result)
% model_type='' or 'q', 'q' is the Natheniel's model

if ~strcmp(model_type,'q')
    load(filepath_result,'filepath_csv','r_pred_all','r_hat_all','lr_all','lik','trials_to_try','params')
else
    load(filepath_result)
end
[folder_result,fn_result,ext]=fileparts(filepath_result);
[~,fn_csv,~]=fileparts(filepath_csv);

T = readtable(filepath_csv);

%% parse stem, leaf choices

obs={};
obs{1}=[or(T.leaf==1,T.leaf==2),T.reward];
obs{2}=[or(T.leaf==3,T.leaf==4),T.reward];
obs{3}=[or(T.leaf==5,T.leaf==6),T.reward];
for l=1:6
    obs{l+3}=[T.leaf==l,T.reward];
end
for o_index=1:length(obs)
    obs{o_index}=obs{o_index}(trials_to_try,:);
end
colors=[0, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
    0.4660, 0.6740, 0.1880;
    0.3010, 0.7450, 0.9330];

%% get contingency (min leaf, max leaf, mean stem reward)
cont_ind={};
cont_ind{1}{1}=[1,2]; cont_ind{1}{2}=[3,4]; 
% first 2 digits are prob for 1st leaf, 2-4 digits are prob for 2nd leaf.
cont_ind{2}{1}=[5,6]; cont_ind{2}{2}=[7,8];
% 5-6 digits are prob for 3rd leaf, 7-8 digits are prob for 4th leaf.
cont_ind{3}{1}=[9,10]; cont_ind{3}{2}=[11,12];

cont_ind{4}{1}=[1,2]; cont_ind{4}{2}=cont_ind{4}{1};
cont_ind{5}{1}=[3,4]; cont_ind{5}{2}=cont_ind{5}{1};
cont_ind{6}{1}=[5,6]; cont_ind{6}{2}=cont_ind{6}{1};

cont_ind{7}{1}=[7,8]; cont_ind{7}{2}=cont_ind{7}{1};
cont_ind{8}{1}=[9,10]; cont_ind{8}{2}=cont_ind{8}{1};
cont_ind{9}{1}=[11,12]; cont_ind{9}{2}=cont_ind{9}{1};

contingency=T.contingency;
session=T.session;
cont_str=num2str(T.contingency(trials_to_try));
cont={};
for cont_i=1:9
    cont{cont_i}=zeros(length(trials_to_try),3);
    prob1=str2num(cont_str(:,cont_ind{cont_i}{1}));
    prob2=str2num(cont_str(:,cont_ind{cont_i}{2}));
    cont{cont_i}(:,2)=min([prob1,prob2],[],2)/100;
    cont{cont_i}(:,3)=max([prob1,prob2],[],2)/100;
    cont{cont_i}(:,1)=1/2*(cont{cont_i}(:,2)+cont{cont_i}(:,3));
end
%% get r_pred_all, r_hat_all, and calculate likelihood
if strcmp(model_type,'q')
    r_hat_all=zeros(9,length(trials_to_try));
    r_pred_all=zeros(9,length(trials_to_try));
    lr_all=zeros(9,length(trials_to_try));

    t_count=0;
    for s_index=1:numel(Q) % sessions
        for t=1:(size(Q{s_index},3)-1)
            r_pred_all(4:9,t_count+t)=(reshape(Q{s_index}(:,:,t+1)',6,1)+1)/2; %the scaling here is because Nathiel did the scaling, here undoing it.
            %scale it so that it lies between 0 and 1
        end
        t_count=t_count+size(Q{s_index},3)-1;
        
        lr_all(1:9,t_count:(t_count+size(Q{s_index},3)-2))=repmat(lr(s_index),9,size(Q{s_index},3)-1); 
            %scale it so that it lies between 0 and 1
        t_count=t_count+size(Q{s_index},3);
    end
    for o_index=1:3
        r_hat_all(o_index,:)=0.5*(r_hat_all((o_index+1)*2,:)+r_hat_all((o_index+1)*2+1,:));
        %lr_all(o_index,:)=0.5*(lr_all((o_index+1)*2,:)+lr_all((o_index+1)*2+1,:));
        r_pred_all(o_index,:)=0.5*(r_pred_all((o_index+1)*2,:)+r_pred_all((o_index+1)*2+1,:));
    end
    
    stem_choice=floor(T.leaf/2)+mod(T.leaf,2);
    leaf_choice=T.leaf;
    
    boundary=find(diff([0;session]));
    lik=[];
    r_pred_modified_all=[];
    for b=1:length(boundary)
        if and(boundary(b)>=min(trials_to_try),boundary(b)<=max(trials_to_try))
            session_begin=boundary(b);
            session_end=min([boundary(b+1),max(trials_to_try)]);
            
            stem_choices=stem_choice(session_begin:session_end);
            leaf_choices=leaf_choice(session_begin:session_end);
            Q_leaf=r_pred_all(4:9,session_begin:session_end);
            Q_stem=r_pred_all(1:3,session_begin:session_end);
            
            [lik_,bias,beta_stem,beta_leaf,q_stem_modified,q_leaf_modified]=...
                loglikelihood_beta(Q_stem,Q_leaf,stem_choices,leaf_choices);
            lik=[lik,lik_];
            r_pred_modified=[q_leaf_modified;q_stem_modified];
            r_pred_modified_all=[r_pred_modified_all,r_pred_modified];
        end
    end
else
    
end
%%
page_num=ceil(length(trials_to_try)/300);

for plot_ind=1:page_num
    trial_start_index=300*(plot_ind-1)+1;
    trial_end_index=min([300*(plot_ind),length(trials_to_try)]);
    plot_range{plot_ind}=trials_to_try(trial_start_index:trial_end_index);
    f=figure('units','normalized','outerposition',[0 0 1 1]);
    %f.Position = [10 10 1000 600];
    
    axe_cells=cell(1,2);
    titles={'stem1','stem 2','stem 3'};
    axe_cells{1}=subplot(3,1,1);

% plot
for o_index=1:3
    
    observations=obs{o_index}(trial_start_index:trial_end_index,:);

    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    
    r_hat=r_hat_all(o_index,trial_start_index:trial_end_index);
    r_pred=r_pred_all(o_index,trial_start_index:trial_end_index);
    r_pred_modified=r_pred_modified_all(o_index,trial_start_index:trial_end_index);
     
    rewarded=double(observations(:,2)>0);
    rewarded(rewarded==0)=nan;
    unrewarded=double(observations(:,2)<=0);
    unrewarded(unrewarded==0)=nan;
    observations((observations(:,1)==0),1)=nan;
    
    plot(r_pred_modified,'color',c1,'LineWidth',1)
    hold on
    plot(r_pred,'color',[c1,0.35],'LineWidth',2)
    rewarded_choice=r_hat.*observations(:,1)'.*rewarded';
    unrewarded_choice=r_hat.*observations(:,1)'.*unrewarded';
    plot(rewarded_choice,'+','color',c1,'MarkerSize',5)
    plot(unrewarded_choice,'o','color',c1,'MarkerSize',5)
    
end
axis tight
ylim([0,1])
xticks(1:100:length(plot_range{plot_ind}))
xticklabels(plot_range{plot_ind}(1:100:length(plot_range{plot_ind})))
title('inferred reward rate')

axe_cells{2}=subplot(3,1,2);
for o_index=1:3
    c1=colors(o_index,:);
    lr=lr_all(o_index,:);
    plot(lr(trial_start_index:trial_end_index),'color',c1)
    hold on
end
axis tight
ylim([0,1])
xticks(1:100:length(plot_range{plot_ind}))
xticklabels(plot_range{plot_ind}(1:100:length(plot_range{plot_ind})))
title('inferred learning rate')

axe_cells{3}=subplot(3,1,3);
for o_index=1:3
    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    x_range=1:length(plot_range{plot_ind});
    fill([x_range fliplr(x_range)],...
        [cont{o_index}(trial_start_index:trial_end_index,2)' fliplr(cont{o_index}(trial_start_index:trial_end_index,3)')],c1,'EdgeColor','None','FaceAlpha',0.5)
    hold on
end
for o_index=1:3
    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    plot(cont{o_index}(trial_start_index:trial_end_index,1),':','color',c2,'LineWidth',3)
    hold on
end
axis tight
ylim([0,1])
xticks(1:100:length(plot_range{plot_ind}))
xticklabels(plot_range{plot_ind}(1:100:length(plot_range{plot_ind})))
title('ground truth contingency')

for axe_index=1
    axes(axe_cells{axe_index});

    % plot session boundaries
    boundary=find(diff([0;session]));
    b_range=find(boundary>=min(trials_to_try),1):find(boundary<=max(trials_to_try),1,'last');
    for b_index=1:length(b_range)
        b=b_range(b_index);
        if and(boundary(b)>=min(plot_range{plot_ind}),boundary(b)<=max(plot_range{plot_ind}))
            bound=boundary(b)-min(plot_range{plot_ind})+1;
            plot([bound,bound],[0,1],'color',[0.8,0.1,0.1])
            if ~strcmp(model_type,'q')
                h1=text(bound,0,num2str(lik(b_index)));
                h2=text(bound+10,0,['bias_stay',sprintf('%.1f',round(params(b_index,1),1))]);
                h3=text(bound+20,0,['beta_stem',sprintf('%.1f',round(params(b_index,2),1))]);
                h4=text(bound+30,0,['beta_leaf',sprintf('%.1f',round(params(b_index,3),1))]);
                %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
                set(h1,'Rotation',90);
                set(h2,'Rotation',90);
                set(h3,'Rotation',90);
                set(h4,'Rotation',90);
            end
        end
    end
end

lgd=legend(axe_cells{1},'beta stem 1','alpha stem 1','rewarded choice','unrewarded choice','beta stem 2','alpha stem 2','rewarded choice','unrewarded choice','beta stem 3','alpha stem 3','rewarded choice','unrewarded choice');
lgd.FontSize = 5;
hleg = legend('show');
% legend_to_plot=hleg.String(1:6);
% hleg.String = legend_to_plot;

legend(axe_cells{2},'stem 1','stem 2','stem 3')
hleg = legend('show');
legend_to_plot=hleg.String(1:3);
hleg.String = legend_to_plot;

legend(axe_cells{3},'stem 1','stem 2','stem 3','stem 1 mean','stem 2 mean','stem 3 mean')
hleg = legend('show');
legend_to_plot=hleg.String(1:3);
hleg.String = legend_to_plot;

h=suptitle(fn_csv);
h.Interpreter = 'none';

linkaxes([axe_cells{1},axe_cells{2},axe_cells{3}])

%save(fullfile(folder_result,[fn_result,'.pdf']))
orient(f,'landscape')
print(f,fullfile(folder_result,[fn_result,'_p',num2str(plot_ind)]),'-dpdf')
close(f)
end
%%
% f=figure;
% f.Position = [10 10 1000 600];
% axe_cells=cell(1,2);
% titles={'leaf1','leaf2','leaf3','leaf4','leaf5','leaf6'};
% axe_cells{1}=subplot(3,1,1);
% 
% % plot
% for o_index=4:9
%     observations=obs{o_index};
%     c1=colors(o_index-3,:);
%     c2=min(colors(o_index-3,:).*[2,2,2],[1,1,1]);
%     
%     r_hat=r_hat_all(o_index,:);
%     %r_hat=fillmissing(r_hat,'previous');
%     %plot(r_hat,'.','color',c1)
%     %hold on
%     
%     rewarded=double(observations(1:trials_to_try,2)>0);
%     rewarded(rewarded==0)=nan;
%     unrewarded=double(observations(1:trials_to_try,2)<=0);
%     unrewarded(unrewarded==0)=nan;
%     
%     plot(r_hat.*rewarded','.','color',c1,'MarkerSize',5)
%     hold on
%     plot(r_hat.*unrewarded','o','color',c1,'MarkerSize',2)
% end
% axis tight
% ylim([0,1])
% 
% title('inferred reward rate')
% 
% axe_cells{2}=subplot(3,1,2);
% for o_index=4:9
%     c1=colors(o_index-3,:);
%     observations=obs{o_index};
%     
%     r_hat=r_hat_all(o_index,:);
%     notnan_index=~isnan(r_hat);
%     lr=lr_all(o_index,:);
%     plot(find(notnan_index),lr(notnan_index),'color',c1)
%     hold on
% end
% axis tight
% ylim([0,1])
% title('inferred learning rate')
% 
% axe_cells{3}=subplot(3,1,3);
% for o_index=1:3
%     c1=colors(o_index,:);
%     c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
%     fill([1:trials_to_try fliplr(1:trials_to_try)],...
%         [cont{o_index}(:,2)' fliplr(cont{o_index}(:,3)')],c1,'EdgeColor','None','FaceAlpha',0.5)
%     hold on
% end
% for o_index=1:3
%     c1=colors(o_index,:);
%     c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
%     plot(cont{o_index}(:,1),':','color',c2,'LineWidth',3)
%     hold on
% end
% axis tight
% ylim([0,1])
% title('ground truth contingency')
% 
% for axe_index=1:2
%     axes(axe_cells{axe_index});
% %   not applicable to depletion contingency
% %     % plot contingency boundaries
% %     boundary=find(diff([0;contingency]));
% %     for b=1:length(boundary)
% %         if boundary(b)<max(find(1:trials_to_try))
% %             plot([boundary(b),boundary(b)],[0,1],'color',[0.1,0.1,0.1])
% %             %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
% %             %set(h,'Rotation',90);
% %         end
% %     end
%     % plot session boundaries
%     boundary=find(diff([0;session]));
%     for b=1:length(boundary)
%         if boundary(b)<max(find(1:trials_to_try))
%             plot([boundary(b),boundary(b)],[0,1],'color',[0.8,0.1,0.1])
%             if ~strcmp(model_type,'q')
%                 h=text(boundary(b),0,num2str(lik_leaf(b)));
%             %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
%             else
%                 h=text(boundary(b),0,num2str(lik(b)));
%             end
%             set(h,'Rotation',90);
%         end
%     end
% end
% 
% legend(axe_cells{1},'leaf 1','','leaf 2','','leaf 3','','leaf 4','','leaf 5','','leaf 6','')
% hleg = legend('show');
% %legend_to_plot=hleg.String(1:end);
% %hleg.String = legend_to_plot;
% 
% legend(axe_cells{2},'leaf 1','leaf 2','leaf 3','leaf 4','leaf 5','leaf 6')
% hleg = legend('show');
% legend_to_plot=hleg.String(1:6);
% hleg.String = legend_to_plot;
% 
% legend(axe_cells{3},'stem 1','stem 2','stem 3','stem 1 mean','stem 2 mean','stem 3 mean')
% hleg = legend('show');
% legend_to_plot=hleg.String(1:3);
% hleg.String = legend_to_plot;
% 
% h=suptitle(csv_file);
% h.Interpreter = 'none';