function results2=count_contingency(T)
%%%
% Returns results2
% For each contingency where #20, #50, #80 == #321 or #411,
% count the number of occurences of the exact contingency

% Reduced contingency isn't being used
% Nor is theo_count
%%%

%% contingency summary
cont=T.contingency;
cont_u=unique(cont);
cont_hist=zeros(1,length(cont_u));
counts=containers.Map(); %reduced - based on #20, #50, #80
counts2=containers.Map(); %nonreduced - full contingency, but only for #321 and #411

for i=1:length(cont_u)
    current_cont=cont_u(i);
    current_cont_n=sum(current_cont==cont);
    
    current_cont=num2str(current_cont);
    % Leading 0 may be dropped; check if string is too short
    if length(current_cont)<12
        current_cont=['0',current_cont];
    end
    current_cont_full=current_cont;
    
    % Turn string "202080205080" into [20 20 80 20 50 80] (doubles)
    current_cont=[str2double(current_cont(1:2)),str2double(current_cont(3:4)),str2double(current_cont(5:6)),...
        str2double(current_cont(7:8)),str2double(current_cont(9:10)),str2double(current_cont(11:12))];
    % Find the number of instances of '20', '50', and '80'
    current_cont=[sum(current_cont==20),sum(current_cont==50),sum(current_cont==80)];
    % Turn into a string of 'abc' where a=#20, b=#50, c=#80
    current_cont=squeeze(char(string(current_cont)))';
    
    if ismember(current_cont,{'321','411'})
        counts2(current_cont_full)=current_cont_n;
    end
    
    % If current set of probs is new, add a new entry.
    % However, there may be permutations of the probabilities
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

conts2=keys(counts2);
cont_ns2=values(counts2);

[sorted_counts2,I]=sort(cell2mat(cont_ns2),'descend');
most_cont2=char(conts2(I));

%% theoretical counts
theo_count=zeros(11,1);
for r=1:11
    theo_count(r)=nchoosek(6,str2double(most_cont(r,1)))*nchoosek(6-str2double(most_cont(r,1)),str2double(most_cont(r,2)));
end
theo_count=round(theo_count./sum(theo_count)*length(cont));

%% write table: results and results2 are the two major outputs
numof20=most_cont(:,1);
numof50=most_cont(:,2);
numof80=most_cont(:,3);
results=table(numof20,numof50,numof80,sorted_counts');
results.Properties.VariableNames = {'numof20','numof50','numof80','counts'};

results2=table(most_cont2,sorted_counts2');
results2.Properties.VariableNames = {'contingency','counts'};

%% test code, ignore for now
% cont_temp=[20,20,20,20,50,80];
% P = unique(perms(cont_temp),'row');
% summation=0;
% sums=[];
% for i=1:length(P)
%     Pi=string(P(i,1))+string(P(i,2))+string(P(i,3))+string(P(i,4))+string(P(i,5))+string(P(i,6));
%     current_cont_n=sum(Pi==cont);
%     sums=[sums;current_cont_n];
%     summation=summation+current_cont_n;
% end

end