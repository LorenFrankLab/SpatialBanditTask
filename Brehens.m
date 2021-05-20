%% load data

filepath='/Users/shijiegu/Documents/Second-Year/w_Alison';
csv_file='senor_clean_contingencies_only_parsed_data.csv';
T = readtable(fullfile(filepath,csv_file));
session=T.session;
contingency=T.contingency;
trials_to_try=720;

%% Tim Brehens 2007
number=50;
r=linspace(0.01,0.99,number); % avoid 0 and 1 as these can give Inf in beta distribution
dr=r(2)-r(1);
v=linspace(-4,log(1/2),number);
dv=v(2)-v(1);
K=0.1;

obs={};
obs{1}=[or(T.leaf==1,T.leaf==2),T.reward];
obs{2}=[or(T.leaf==3,T.leaf==4),T.reward];
obs{3}=[or(T.leaf==5,T.leaf==6),T.reward];
for l=1:6
    obs{l+3}=[T.leaf==l,T.reward];
end
alphas={};
betas={};
%%
parfor o_index=1:len(obs)
    observations=obs{o_index}(:,1);
    reward=obs{o_index}(:,2);
    [alpha,beta]=BrehensModel(observations,reward,K,r,dr,v,dv);
    alphas{o_index}=alpha;
    betas{o_index}=beta;
end

%%
figure;

obs{1}=[or(T.leaf==1,T.leaf==2),T.reward];
obs{2}=[or(T.leaf==3,T.leaf==4),T.reward];
obs{3}=[or(T.leaf==5,T.leaf==6),T.reward];
obs{4}=[T.leaf==1,T.reward];

titles={'leaf1','stem 2','stem 3','stem 1'};
for o_index=1:3
    subplot(4,1,o_index)
    observations=obs{o_index};
    alpha=alphas{o_index};
    beta=betas{o_index};
    
    
    r_hat=sum(r*squeeze(sum(alpha(:,:,1:trials_to_try),2)),1);
    r_pred=sum(r*squeeze(sum(beta(:,:,1:trials_to_try),2)),1);
    lr=(diff([0,r_hat]))./(observations(1:trials_to_try,2)'-r_hat(1:end));
    %lr=(r_pred(2:end)-r_hat(2:end));
    plot(lr)
    hold on
    plot(r_hat)
    
    
    title(titles{o_index})
    
    % plot contingency boundaries
    boundary=find(diff([0;contingency]));
    for b=1:length(boundary)
        if boundary(b)<max(find(1:trials_to_try))
            plot([boundary(b),boundary(b)],[0,1],'color',[0.1,0.1,0.1])
            h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
            set(h,'Rotation',90);
        end
    end
    % plot session boundaries
    boundary=find(diff([0;session]));
    for b=1:length(boundary)
        if boundary(b)<max(find(1:trials_to_try))
            plot([boundary(b),boundary(b)],[0,1],'color',[0.8,0.1,0.1])
            h=text(boundary(b),0.5,num2str(T.contingency(boundary(b))));
            set(h,'Rotation',90);
        end
    end
    
    if o_index==1
        legend('learning rate','estimated reward rate')
        hleg = legend('show');
        legend_to_plot=hleg.String(1:2);
        hleg.String = legend_to_plot;
    end

    axis tight
    
    ylim([0,1])
end


%%
function [p,a,b]=p_r_transition(x,mean,width,dr,r)
    p=0;
    a=1/width*mean;
    b=1/width-a;
    if and(a>0,b>0)
        p=betapdf(x,a,b)*dr/sum(betapdf(r,a,b)*dr);
    end
    if isinf(p)
        p=0;
    end
end