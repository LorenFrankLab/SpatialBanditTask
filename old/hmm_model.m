filepath='/home/shijiegu/Documents/w_Alison';
animal='senor';
depletion_flag=1;

% csv_file to estimate value for
if depletion_flag
    csv_file=[animal, '_clean_contingencies_only_parsed_depletion_data.csv'];
    mkdir(fullfile(filepath,['hmm_decay_',animal]))
else
    csv_file=[animal, '_clean_contingencies_only_parsed_data.csv'];  
    mkdir(fullfile(filepath,['hmm_',animal]))
end
T = readtable(fullfile(filepath,'data',csv_file));


%% hmm params (1): Emission prob
results2=count_contingency(T);

numStates=size(results2.contingency,1);
numObs=6;
states=zeros(size(numStates,1),numObs);
for si=1:size(results2.contingency,1)
    for ai=1:numObs
        states(si,ai)=str2num(results2.contingency(si,2*ai-1:2*ai))/100;
    end
end

% states = ones(6,6)*0.2;
% states = full(spdiags(ones(6,1).*0.8,0,states));
%% hmm params (2): Transition prob

v=1-1/30;   %volatiliy

% Transition prob
transmat = ones(numStates,numStates)*(1-v)/(numStates-1); %to others
transmat = full(spdiags(ones(numStates,1).*v,0,transmat)); %to self

%%

dates=unique(T.date);

for d=1:length(dates)
    day_index=T.date==dates(d);
    legal_trial_ind_day=[];
    entropy=[];
    lr=[];
    Q=[];
    alpha=[];
    sessions=unique(T(day_index,:).session);
    for si=1:length(sessions)
        s=sessions(si);
        day_session_index=(day_index.*(T.session==s))>0;
        T_short=T(day_session_index,:);
        unique_contingencies=unique(T_short.contingency,'stable');
        
        legal_trial_ind=find(day_session_index.*(T.contingency~=unique_contingencies(1)));
        legal_trial_ind_day=[legal_trial_ind_day;legal_trial_ind];
        
        %% load data
        
        choice=T_short.leaf;
        reward=T_short.reward;
        
        %% decode hmm by forward algorithm
        
        % initialization
        alpha_=ones(numStates,size(T_short,1)+1)./numStates;
        emission_prob=reward(1)*states(:,choice(1))+(1-reward(1))*(1-states(:,choice(1)));
        alpha_(:,2)=emission_prob;
        alpha_(:,2)=alpha_(:,2)./sum(alpha_(:,2));
        
        % for depletion onlt
        success_leave=zeros(1,6);
        for i=2:size(T_short,1)
            if depletion_flag
                success_leave(choice(i)) = success_leave(choice(i))+1; % successive leave visit
                success_leave(setdiff(1:6,[ceil(choice(i)/2)*2,ceil(choice(i)/2)*2-1])) = 0;
                states_=bsxfun(@times,states,0.8.^(success_leave));
                emission_prob=reward(i)*states_(:,choice(i))+(1-reward(i))*(1-states_(:,choice(i)));
            else
                emission_prob=reward(i)*states(:,choice(i))+(1-reward(i))*(1-states(:,choice(i)));
            end

            alpha_(:,i+1)=(alpha_(:,i)'*transmat).*emission_prob';
            alpha_(:,i+1)=alpha_(:,i+1)./sum(alpha_(:,i+1));
        end
        entropy_ = -sum(alpha_.*log2(alpha_),1);
        Q_=states'*alpha_;
        lr_=zeros(1,size(T_short,1));
        for i=1:size(T_short,1)
            Q_old=Q_(choice(i),i);
            Q_new=Q_(choice(i),i+1);
            lr_(i)=(Q_new-Q_old)/(reward(i)-Q_old);
        end
        
        entropy=[entropy,entropy_(1:size(T_short,1))];
        Q=[Q,Q_(:,1:size(T_short,1))];
        alpha=[alpha,alpha_(:,1:size(T_short,1))];
        lr=[lr,lr_];
        
        %% plot examples
%         try
%         for b=1:2
%             figure;
%             subplot(2,1,1)
%             imagesc(alpha(:,(60*(b)+1):60*(b+1)))
%             hold on;
%             choice_=T_short.leaf((60*(b)+1):60*(b+1));
%             reward_=T_short.reward((60*(b)+1):60*(b+1));
%             reward_(reward_==0)=NaN;
%             plot(1:60,choice_,'*','color',[1,1,1])
%             plot(1:60,choice_.*reward_,'+','color',[0.8,0.2,0.2])
%             subplot(2,1,2)
%             plot(1:60,entropy((60*(b)+1):60*(b+1)));
%             title(string(T_short.contingency(60*(b)+1)))
%             print(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(d),'session',num2str(s),'_example','.pdf']),'-dpdf')
%             close
%         end
%         catch
%         end
    end
        
    %seq=choice.*reward+1;
    %[PSTATES,logpseq,FORWARD,BACKWARD,S] = hmmdecode(seq',transmat,emissionprob);
    %% save "Q values"
    if depletion_flag
        save(fullfile(filepath,['hmm_decay_',animal],[animal,'_day',num2str(d),'_q_value.mat']),'alpha','Q','lr','entropy','-v7.3')
    else
        save(fullfile(filepath,['hmm_',animal],[animal,'_day',num2str(d),'_q_value.mat']),'alpha','Q','lr','entropy','-v7.3')
    end
end