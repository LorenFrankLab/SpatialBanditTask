animal='senor';
load(['/home/shijiegu/Documents/w_Alison/hmm_',animal,'/',animal,'_summary.mat'])
load(['/home/shijiegu/Documents/w_Alison/q_learning_',animal,'/',animal,'_summary.mat'])
%load('/home/shijiegu/Documents/w_Alison/q_learning_senor/senor_summary.mat')
figure;
for daynum=1:size(iaic_hmm,1)
    subplot(size(iaic_hmm,1),1,daynum)

    plot(1:size(iaic_hmm,2),iaic_hmm(daynum,:),'x-')
    hold on
    plot(1:size(iaic_q_learning,2),iaic_q_learning(daynum,:),'x-')

    xticks(1:5)
    xticklabels({'full','nogo','old','nogonobias','null'})
    ylabel('AIC')
    if daynum==1
        legend({'hmm','q'})
    end
end
suptitle(animal)
%%
animal='peanut';
load(['/home/shijiegu/Documents/w_Alison/hmm_',animal,'/',animal,'_summary.mat'])
load(['/home/shijiegu/Documents/w_Alison/q_learning_',animal,'/',animal,'_summary.mat'])
figure;
for daynum=1:size(iaic_hmm,1)
    plot(1:2,[0,iaic_hmm(daynum,1)-iaic_q_learning(daynum,1)],'x-','color',[0.1,0.08*daynum,0.5])
    hold on
    legends{daynum}=['day ',num2str(daynum)];
end
legend(legends)
xticks([1,2])
xticklabels({'0','hmm-qlearning'})
ylabel('delta AIC')
suptitle(animal)
%% plot parameter: learning rate
animal='senor';
params=readmatrix(['/home/shijiegu/Documents/w_Alison/q_learning_',animal,'/',animal,'_parameter_summary.csv']);
senor_lr=0.5 + 0.5 * erf(params(:,5)/ sqrt(2));

animal='peanut';
params=readmatrix(['/home/shijiegu/Documents/w_Alison/q_learning_',animal,'/',animal,'_parameter_summary.csv']);
peanut_lr=0.5 + 0.5 * erf(params(:,5)/ sqrt(2));

figure;
histogram(senor_lr);
hold on;
histogram(peanut_lr);

%% plot parameter: spatial biase
animals={'senor','peanut'};
figure
for a=1:2
    subplot(1,2,a)
    animal=animals{a};
    params=readmatrix(['/home/shijiegu/Documents/w_Alison/hmm_',animal,'/',animal,'_parameter_summary.csv']);
    spatial_biase=params(:,6:8);
    for daynum=1:size(spatial_biase,1)
        % if fit different parameters for each stem. The minus 1 in the
        %   middle is necessary. All positive values mean L biase?
        % plot(1:3,spatial_biase(daynum,:).*[1,-1,1],'-x','color',[0.1,0.08*daynum,0.5])
        plot(daynum,spatial_biase(daynum,1),'-x','color',[0.1,0.08*daynum,0.5])
        hold on
    legends{daynum}=['day ',num2str(daynum)];
    end
    legend(legends)
    xticks([1,2,3])
    xticklabels({'at stem 1','at stem 2','at stem 3'})
    ylabel('spatial bias')
    title(animal)
end

