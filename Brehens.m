%% load data

csv_folder='/home/shijiegu/Documents/SpatialBanditTask';
result_folder='/home/shijiegu/Documents/SpatialBanditTask/result_Jun172021/trial_5734_6900/';

csv_files={};
csv_files{2}='senor_clean_contingencies_only_parsed_depletion_data.csv';
csv_files{1}='senor_clean_contingencies_only_parsed_data.csv';

trials_to_try=5734:5834;%6900;
for vi=1%:4
    model_version=vi;
for d=1%:numel(csv_files)
    csv_file=csv_files{d};
    result_file=csv_file;
    result_file(end-2:end)='mat';
    result_file=['ver_',num2str(model_version),'_',result_file];
    
    filepath_csv=fullfile(csv_folder,csv_file);
    filepath_result=fullfile(result_folder,result_file);
    
    [alphas,betas,lr_all,r_pred_all,r_hat_all,lik,...
        boundary]=runBrehens(filepath_csv,filepath_result,...
        model_version,trials_to_try);
    
    plot_model('',filepath_result)
end
end
%%
colors=[0, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.4940, 0.1840, 0.5560;
    0.4660, 0.6740, 0.1880;
    0.3010, 0.7450, 0.9330];


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
    plot(r_pred,':','color',c1)
    hold on
    plot(r_hat,'color',c1)
    hold on
    trial_index=find(stem_choice==o_index);
    h=plot(trial_index,zeros(1,length(trial_index))+1,'*','color',c1);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
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
legend(axe_cells{1},'beta stem 1','alpha stem 1','beta stem 2','alpha stem 2','beta stem 3','alpha stem 3')
hleg = legend('show');
legend_to_plot=hleg.String(1:6);
hleg.String = legend_to_plot;

legend(axe_cells{2},'stem 1','stem 2','stem 3','stem 1 mean','stem 2 mean','stem 3 mean')
hleg = legend('show');
legend_to_plot=hleg.String(1:6);
hleg.String = legend_to_plot;
