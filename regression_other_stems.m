%% regression current value

%% load behavior data
filepath='/home/shijiegu/Documents/w_Alison';
animal='peanut';
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


for daynum=1:min([10,length(dates)])
    figure;
    set(gcf,'Position',[100 100 1500 800])
    sessions_day=sessions(T.date==dates(daynum));
    unique_sessions=unique(sessions_day);
    
    for sesnum=1:min([length(unique_sessions),8])
        subplot(2,4,sesnum);
        
        session_ind=unique_sessions(sesnum);

        %% load and parse value data
        %daynum=1;
        load(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))
        %load(fullfile(filepath,['q_learning_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))

        %% parse behavior data
        
        stem_switch_day=stem_switch_all(T.date==dates(daynum));
        stem_switch_time_day=find(stem_switch_day);
        stem_switch_time_day=stem_switch_time_day(stem_switch_time_day>1);
        time_session=find(sessions_day==session_ind);
        stem_switch_time=intersect(time_session,stem_switch_time_day);
         
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
                X(t_ind)=X2-X1;
            else
                X(t_ind)=X1-X2;
            end
        end

        %%
        X_full=X;
        Y_full=Y;
        mdl=fitglm(X_full,Y_full,'linear','Distribution','binomial');

        %figure;
        scatter(X_full,Y_full)
        hold on;
        Y_hat=1./(1+exp(-table2array(mdl.Coefficients(2,1)).*X_full-table2array(mdl.Coefficients(1,1))));
        scatter(X_full,Y_hat)
        [sortedx,ind]=sort(X_full,'descend');
        plot(sortedx,smoothdata(Y_full(ind),'movmean',10))
        xlabel('value of the two stems')
        ylabel('choose')
        title({['day number:',num2str(daynum)],['beta ',num2str(table2array(mdl.Coefficients(2,'Estimate'))),' p value ',num2str(round(table2array(mdl.Coefficients(2,'pValue')),5))],['SSE ',num2str(mdl.SSE),' p value, ', num2str(round(mdl.devianceTest.pValue(2),5))]})
        
        set(gcf, 'PaperUnits', 'inches');
        set(gcf, 'PaperSize', [10 5]);
        %print('-bestfit',fullfile(filepath,['q_learning_',animal],[animal,'_day',num2str(daynum),'_regression_other_stem_value']),'-dpdf')
        print('-bestfit',fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_regression_other_stem_value']),'-dpdf')
    end
end
%%