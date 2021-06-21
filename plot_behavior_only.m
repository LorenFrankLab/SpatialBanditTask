csv_folder='/home/shijiegu/Documents/SpatialBanditTask';
csv_files='senor_clean_contingencies_only_parsed_depletion_data.csv';
filepath_csv=fullfile(csv_folder,csv_file);
T = readtable(filepath_csv);

trials_to_try=5000;
obs={};
obs{1}=[or(T.leaf==1,T.leaf==2),T.reward];
obs{2}=[or(T.leaf==3,T.leaf==4),T.reward];
obs{3}=[or(T.leaf==5,T.leaf==6),T.reward];
for l=1:6
    obs{l+3}=[T.leaf==l,T.reward];
end
for o_index=1:length(obs)
    obs{o_index}=obs{o_index}(1:trials_to_try,:);
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
cont_str=num2str(T.contingency(1:trials_to_try));
cont={};
for cont_i=1:9
    cont{cont_i}=zeros(trials_to_try,3);
    prob1=str2num(cont_str(:,cont_ind{cont_i}{1}));
    prob2=str2num(cont_str(:,cont_ind{cont_i}{2}));
    cont{cont_i}(:,2)=min([prob1,prob2],[],2)/100;
    cont{cont_i}(:,3)=max([prob1,prob2],[],2)/100;
    cont{cont_i}(:,1)=1/2*(cont{cont_i}(:,2)+cont{cont_i}(:,3));
end
%%
figure;

for o_index=1:3
    c1=colors(o_index,:);
    c2=min(colors(o_index,:).*[2,2,2],[1,1,1]);
    x_range=1:trials_to_try;
    fill([x_range fliplr(x_range)],...
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
%xticks(1:100:length(plot_range{plot_ind}))
%xticklabels(plot_range{plot_ind}(1:100:length(plot_range{plot_ind})))
title('ground truth contingency')




    % plot session boundaries
    boundary=find(diff([0;session]));
    for b=1:length(boundary)
        if boundary(b)<=trials_to_try
            bound=boundary(b);
            plot([bound,bound],[0,1],'color',[0.8,0.1,0.1])
            if ~strcmp(model_type,'q')
                h1=text(bound,0,num2str(lik(b)));
                %h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
                set(h1,'Rotation',90);
            end
        end
    end
