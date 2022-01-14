%% contingency summary
cont_u=unique(cont);
cont_hist=zeros(1,length(cont_u));
counts=containers.Map();

for i=1:103
    current_cont=cont_u(i);
    current_cont_n=sum(current_cont==cont);
    current_cont=char(current_cont);
    current_cont=[str2double(current_cont(1:2)),str2double(current_cont(3:4)),str2double(current_cont(5:6)),...
        str2double(current_cont(7:8)),str2double(current_cont(9:10)),str2double(current_cont(11:12))];
    current_cont=[sum(current_cont==20),sum(current_cont==50),sum(current_cont==80)];
    current_cont=squeeze(char(string(current_cont)))';
    if ismember(current_cont,keys(counts))
        counts(current_cont)=counts(current_cont)+current_cont_n;
    else
        counts(current_cont)=current_cont_n;
    end
end

conts=keys(counts);
cont_ns=values(counts);

[sorted_counts,I]=sort(cell2mat(cont_ns),'descend');
most_cont=char(conts(I));

%% theoretical counts
theo_count=zeros(11,1);
for r=1:11
    theo_count(r)=nchoosek(6,str2double(most_cont(r,1)))*nchoosek(6-str2double(most_cont(r,1)),str2double(most_cont(r,2)));
end
theo_count=round(theo_count./sum(theo_count)*length(cont));

%% write table
numof20=most_cont(:,1);
numof50=most_cont(:,2);
numof80=most_cont(:,3);
results=table(numof20,numof50,numof80,sorted_counts',theo_count);
results.Properties.VariableNames = {'numof20','numof50','numof80','counts','theoretical count'};

%% test code
cont_temp=[20,20,20,20,50,80];
P = unique(perms(cont_temp),'row');
summation=0;
sums=[];
for i=1:length(P)
    Pi=string(P(i,1))+string(P(i,2))+string(P(i,3))+string(P(i,4))+string(P(i,5))+string(P(i,6));
    current_cont_n=sum(Pi==cont);
    sums=[sums;current_cont_n];
    summation=summation+current_cont_n;
end