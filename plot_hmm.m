filepath='/home/shijiegu/Documents/w_Alison';
animal='peanut';
depletion_flag=1;

if depletion_flag
    csv_file=[animal, '_clean_contingencies_only_parsed_depletion_data.csv'];
else
    csv_file=[animal, '_clean_contingencies_only_parsed_data.csv'];
end
T = readtable(fullfile(filepath,'data',csv_file));
dates=T.date;
dates=unique(dates);
rng(2022)

%% stem switch triggered entropy
% L=10; %window length
% tmp=[0;diff(cell2mat(T(day_index,:).stem))];
% ind=find(tmp~=0); %stem switch times within the day's all data
% ind_abs=find(tmp~=0)+find(day_index,1)-1; %stem switch times in all days
% ind_legal=ismember(ind_abs,legal_trial_ind_day);
% ind=ind(ind_legal); % index within alpha or entropy; within day.
% ind_abs=ind_abs(ind_legal); % index among all days data
% 
% entropies=zeros(length(ind),2*L+1); % place holder
% s_legal_ind=[];
% for sw=2:length(ind)-1 % start with 2, because we want to compare with the one before
%     %if and(and(ind(sw)-L>0,ismember(ind_abs(sw)+L,legal_trial_ind_day)),ind(sw+1)-ind(sw)>L)
%     if and(and(ind(sw)-L>0,ismember(ind_abs(sw)+L,legal_trial_ind_day)),ind(sw)-ind(sw-1)>L)
%         ind1=ind(sw)-L;
%         ind2=ind(sw)+L;
%         entropies(sw,:)=entropy(ind1:ind2);
%         s_legal_ind=[s_legal_ind;sw];
%     end
% end
% entropies=entropies(s_legal_ind,:);
% 
% % for null hypothesis
% entropies_null_mean=[];
% for b=1:1000
%     s_legal_ind=[1];
%     entropies_null=zeros(size(entropies,1),2*L+1);
%     counti=1;
%     while length(s_legal_ind)<=size(entropies,1)
%         ind=randsample(length(legal_trial_ind_day),1);
%         ind_abs=legal_trial_ind_day(ind); % index within all days' data
%         closest_previous_ind=s_legal_ind(find(s_legal_ind<=ind_abs,1,'last'));
%         
%         if and(and(ind_abs-L>0,ismember(ind_abs+L,legal_trial_ind_day)),ind_abs-closest_previous_ind>L)
%             ind1=ind_abs-find(day_index,1)+1-L;
%             ind2=ind_abs-find(day_index,1)+1+L;
%             entropies_null(counti,:)=entropy(ind1:ind2);
%             s_legal_ind=sort([s_legal_ind;ind_abs],'ascend');
%         end
%         counti=counti+1;
%     end
%     y=mean(entropies_null,1);
%     entropies_null_mean=[entropies_null_mean;y];
% end

% %% plot stem switch triggered entropy
% figure;
% 
% subplot(2,1,1)
% plot(entropies')
% hold on;
% x=1:2*L+1;
% xticks(x)
% xticklabels(mat2cell(x-L-1,1,ones(1,length(x))))
% 
% plot([L+1,L+1],[0,2.6],'k')
% ylabel('entropy')
% xlabel('trials')
% 
% axis tight
% 
% subplot(2,1,2)
% %imagesc(entropies)
% y=mean(entropies,1);
% y_err=std(entropies,0,1)/sqrt(size(entropies,1));
% yu = y+y_err;
% yl = y-y_err;
% x=1:2*L+1;
% fill([x fliplr(x)], [yu fliplr(yl)], [.9 .9 .9], 'linestyle', 'none')
% hold on;
% plot(x,y)
% axis tight
% 
% y_null=mean(entropies_null_mean,1);
% y_err=std(entropies_null_mean,0,1)/sqrt(size(entropies_null_mean,1));
% yu_null = y_null+y_err;
% yl_null = y_null-y_err;
% x=1:2*L+1;
% fill([x fliplr(x)], [yu_null fliplr(yl_null)], [.9 .9 .9], 'linestyle', 'none')
% plot(x,y_null)
% 
% plot([L+1,L+1],[0,2.6],'k')
% xticks(x)
% xticklabels(mat2cell(x-L-1,1,ones(1,length(x))))
% ylabel('entropy')
% xlabel('trials')
% legend('SEM','entropy','SEM','boot random entropy','Location','southeast')
% axis tight
% title({[animal '_day' num2str(d)],'stem_switch_triggered_entropy'},'Interpreter','None')
% print(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(d),'_stem_switch_triggered_entropy_plot_v2.pdf']),'-dpdf')
% close

%% contingency change triggered entropy plot

L=80; %20; %window length
rng(2021);
first_contingency=true;

for daynum=1:length(dates)

    if depletion_flag
        load(fullfile(filepath,['hmm_decay_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))
    else
        load(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))
    end

T_short=T(T.date==dates(daynum),:);
cont_diff=[0;diff(T_short.contingency)];
sess_diff=[0;diff(T_short.session)];

ind=[1;find(sess_diff==1)]; %just in case first trials are zero initialized
alpha(:,ind)=ones(size(alpha,1),length(ind))./size(alpha,1); %just in case first trials are zero initialized
lr(ind)=NaN; %just in case first trials are zero initialized

entropy = -sum(alpha.*log2(alpha),1);

if first_contingency
    ind=[1;intersect(find(cont_diff~=0),find(sess_diff==1))];
else
    % 2nd contingency, 3 contingency
    ind=intersect(find(cont_diff~=0),find(sess_diff==0));
end

lr_matrix=zeros(length(ind),2*L+1);
E_matrix=zeros(length(ind),2*L+1);

ind_legal=[];
for s=1:length(ind) % s for snippet
    if first_contingency
        ind1=ind(s);
        ind2=ind(s)+2*L;
    else
        ind1=ind(s)-L;
        ind2=ind(s)+L;
    end
    
    if and(ind1>=1,ind2<=height(T_short))
        if T_short.session(ind1)==T_short.session(ind2)
            lr_matrix(s,:)=lr(ind1:ind2);
            E_matrix(s,:)=entropy(ind1:ind2);
            ind_legal=[ind_legal;s];
        end
    end
end

lr_matrix=lr_matrix(ind_legal,:);
E_matrix=E_matrix(ind_legal,:);

lr_matrix_null=zeros(length(ind_legal),2*L+1);
E_matrix_null=zeros(length(ind_legal),2*L+1);
% generate null hypothesis
s_count=0;
inds_null=[];
while s_count<length(ind_legal)
    ind_null=randsample(1:height(T_short),1); % need to fix this. 2nd/3rd one should only select from 2nd/3rd cont
    ind1=ind_null-L;
    ind2=ind_null+L;
    
    if and(ind1>1,ind2<=height(T_short))
        if and(T_short.session(ind1)==T_short.session(ind2),sum(ismember([ind1,ind2],find(sess_diff)))==0)
            lr_matrix_null(s_count+1,:)=lr(ind1:ind2);
            E_matrix_null(s_count+1,:)=entropy(ind1:ind2);
            s_count=s_count+1;
            inds_null=[inds_null;ind_null];
        end
    end
end

figure;

subplot(2,2,1)
plot(E_matrix')
hold on
x_=linspace(1,2*L+1,5);
xticks(x_)
if ~first_contingency
    plot([L+1,L+1],[0,2.6],'k')
    xticklabels(mat2cell(x_-L-1,1,ones(1,length(x_))))
else
    xticklabels(mat2cell(x_,1,ones(1,length(x_))))
end

axis tight
ylabel('entropy')
xlabel('trials')
title({[animal '_day' num2str(daynum)],['total trial of ' num2str(length(ind_legal))],'contingency_change_triggered_entropy'},'Interpreter','None')
axis tight

subplot(2,2,2)
plot(lr_matrix')
hold on
x_=linspace(1,2*L+1,5);
xticks(x_)
if ~first_contingency
    plot([L+1,L+1],[0,2.6],'k')
    xticklabels(mat2cell(x_-L-1,1,ones(1,length(x_))))
else
    xticklabels(mat2cell(x_,1,ones(1,length(x_))))
end

axis tight
ylabel('learning rate')
xlabel('trials')
title({[animal '_day' num2str(daynum)],['total trial of ' num2str(length(ind_legal))],'contingency_change_triggered_learning rate'},'Interpreter','None')
axis tight

subplot(2,2,3)
%imagesc(entropies)
y=mean(E_matrix,1);
y_err=std(E_matrix,0,1,'omitnan')/sqrt(length(ind_legal));
yu = fillmissing(y+y_err,'constant',0);
yl = fillmissing(y-y_err,'constant',0);
x=1:2*L+1;
fill([x fliplr(x)], [yu fliplr(yl)], [.9 .9 .9], 'linestyle', 'none')
hold on;
plot(x,y)
axis tight

y_null=mean(E_matrix_null,1);
y_err=std(E_matrix_null,0,1,'omitnan')/sqrt(length(ind_legal));
yu_null = fillmissing(y_null+y_err,'constant',0);
yl_null = fillmissing(y_null-y_err,'constant',0);
fill([x fliplr(x)], [yu_null fliplr(yl_null)], [.9 .9 .9], 'linestyle', 'none','FaceAlpha',0.5)
hold on;
plot(x,y_null)

%plot([L+1,L+1],[0,2.6],'k')
x_=linspace(1,2*L+1,5);
xticks(x_)
if ~first_contingency
    xticklabels(mat2cell(x_-L-1,1,ones(1,length(x_))))
else
    xticklabels(mat2cell(x_,1,ones(1,length(x_))))
end

axis tight
ylabel('entropy')
xlabel('trials')
legend('SEM','entropy','Location','southeast')

title({[animal '_day' num2str(daynum)],'contingency_change_triggered_entropy'},'Interpreter','None')


subplot(2,2,4)
%imagesc(entropies)
y=mean(lr_matrix,1);
y_err=std(lr_matrix,0,1,'omitnan')/sqrt(length(ind_legal));
yu = fillmissing(y+y_err,'constant',0);
yl = fillmissing(y-y_err,'constant',0);
x=1:2*L+1;
fill([x fliplr(x)], [yu fliplr(yl)], [.9 .9 .9], 'linestyle', 'none')
hold on;
plot(x,y)
axis tight
% 
y_null=mean(lr_matrix_null,1);
y_err=std(lr_matrix_null,0,1,'omitnan')/sqrt(length(ind_legal));
yu_null = fillmissing(y_null+y_err,'constant',0);
yl_null = fillmissing(y_null-y_err,'constant',0);
fill([x fliplr(x)], [yu_null fliplr(yl_null)], [.9 .9 .9], 'linestyle', 'none','FaceAlpha',0.5)
plot(x,y_null)

x_=linspace(1,2*L+1,5);
xticks(x_)
if ~first_contingency
    xticklabels(mat2cell(x_-L-1,1,ones(1,length(x_))))
else
    xticklabels(mat2cell(x_,1,ones(1,length(x_))))
end
axis tight
ylabel('learning rate')
xlabel('trials')
%legend('SEM','learning rate','Location','southeast')

title({[animal '_day' num2str(daynum)],'contingency_change_triggered_learning rate'},'Interpreter','None')

if first_contingency
    if depletion_flag
        print(fullfile(filepath,['hmm_decay_',animal],[animal,'_day',num2str(daynum),'_first_contingency_triggered_entropy_plot.pdf']),'-dpdf')
    else
        print(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_first_contingency_triggered_entropy_plot.pdf']),'-dpdf')
    end
else
    print(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_contingency_change_triggered_entropy_plot.pdf']),'-dpdf')
end
close
end