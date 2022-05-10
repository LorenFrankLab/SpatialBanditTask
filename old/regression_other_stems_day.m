%% regression current value

%% load behavior data
filepath='/home/shijiegu/Documents/w_Alison';
animal='senor';
csv_file=[animal, '_clean_contingencies_only_parsed_data.csv'];
T = readtable(fullfile(filepath,'data',csv_file));

stem_=T.stem;
stem1=cellfun(@(x) x=='A',stem_);
stem2=cellfun(@(x) x=='B',stem_);
stem3=cellfun(@(x) x=='C',stem_);
stem=stem1*1+stem2*2+stem3*3;
leaf=T.leaf;

stem_switch_all=[0;diff(stem)>0];

dates=T.date;
dates=unique(dates);
sessions=T.session;


figure;
set(gcf,'Position',[100 100 800 800])
for daynum=1:min([10,length(dates)])
   subplot(5,2,daynum)
   X_full_all=[];
    Y_full_all=[];
    

    load(fullfile(filepath,['q_learning_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))
    %load(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))
    
    stem_switch_day=stem_switch_all(T.date==dates(daynum));
    stem_switch_time=find(stem_switch_day);
    stem_switch_time=stem_switch_time(stem_switch_time>1);
    stem_choice=stem(T.date==dates(daynum));

    
 

        %% regression between behavor and values
        X=zeros(length(stem_switch_time),1);
        Y=zeros(length(stem_switch_time),1);
        for t_ind=1:length(stem_switch_time)
            t=stem_switch_time(t_ind);
            other_stem=setdiff([1,2,3],[stem_choice(t),stem_choice(t-1)]);
            if and(stem_choice(t-1)~=2,other_stem>stem_choice(t))
                Y(t_ind)=1;
            elseif and(stem_choice(t-1)==2,other_stem<stem_choice(t))
                Y(t_ind)=1;
            end
            other_stems=sort([stem_choice(t),other_stem]);
            X1=sum(Q(other_stems(1)*2-1:other_stems(1)*2,t))/2;
            X2=sum(Q(other_stems(2)*2-1:other_stems(2)*2,t))/2;
            if stem_choice(t-1)==2 % current stem
                %X(t_ind,:)=[X2,X1];
                X(t_ind)=X2-X1;
            else
                %X(t_ind,:)=[X1,X2];
                X(t_ind)=X1-X2;
            end
        end
        %%
        X_full=X;
        Y_full=Y;
        X_full_all=[X_full_all;X_full];
        Y_full_all=[Y_full_all;Y_full];

    mdl=fitglm(X_full_all,Y_full_all,'linear','Distribution','binomial');

    %figure;
    scatter(X_full_all,Y_full_all)
    hold on;
    Y_hat = predict(mdl,X_full_all);
    %Y_hat=1./(1+exp(-table2array(mdl.Coefficients(2,1)).*X_full_all-table2array(mdl.Coefficients(1,1))));
    scatter(X_full_all,Y_hat)
    [sortedx,ind]=sort(X_full_all,'descend');
    plot(sortedx,smoothdata(Y_full_all(ind),'movmean',20))
    xlabel('difference of value of the two stems')
    ylabel('choose')
    title({['day number:',num2str(daynum)],...
        ['constant ',num2str(table2array(mdl.Coefficients(1,'Estimate'))),' p value ',num2str(round(table2array(mdl.Coefficients(1,'pValue')),5))],...
        ['beta ',num2str(table2array(mdl.Coefficients(2,'Estimate'))),' p value ',num2str(round(table2array(mdl.Coefficients(2,'pValue')),5))],['SSE ',num2str(mdl.SSE),' p value, ', num2str(round(mdl.devianceTest.pValue(2),5))]})
end    
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [10 10]);

print(fullfile(filepath,['q_learning_',animal],[animal,'_regression_other_stem_value_biase']),'-dpdf')
%print(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_regression_other_stem_value']),'-dpdf')
    

%%