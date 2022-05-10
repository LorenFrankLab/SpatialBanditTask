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

figure;
for daynum=1:min([10,length(dates)])
%% load and parse value data
subplot(5,2,daynum)
load(fullfile(filepath,['q_learning_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))
% if state space model
%load(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(daynum),'_q_value.mat']))

%% parse behavior data
stem_switch=stem_switch_all(T.date==dates(daynum));
stem_choice=stem(T.date==dates(daynum));

%% regression between behavor and values
X=zeros(1,length(stem_choice));
for t=1:length(stem_choice)
    X(t)=sum(Q(stem_choice(t)*2-1:stem_choice(t)*2,t))/2;
end

%%
mdl=fitglm(reshape(X,[],1),reshape(stem_switch,[],1),'linear','Distribution','binomial');

%figure;
scatter(X,stem_switch)
hold on;
Y=1./(1+exp(-table2array(mdl.Coefficients(2,1)).*X-table2array(mdl.Coefficients(1,1))));
scatter(X,Y)
[sortedx,ind]=sort(X,'descend');
plot(sortedx,smoothdata(stem_switch(ind),'movmean',100))
title({['data points:',num2str(length(X))],['day number:',num2str(daynum)],['beta ',num2str(table2array(mdl.Coefficients(2,'Estimate'))),' p value ',num2str(table2array(mdl.Coefficients(2,'pValue')))],['SSE ',num2str(mdl.SSE),' p value, ', num2str(round(mdl.devianceTest.pValue(2),10))]})

end
%%