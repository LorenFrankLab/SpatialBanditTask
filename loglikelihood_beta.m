function [lik,bias,beta_stem,beta_leaf,q_stem_modified,q_leaf_modified]=loglikelihood_beta(q_stem,q_leaf,choice_stem,choice_leaf)

%helper=@(beta) -loglikelihood(beta.*q,choice);

%beta = fminbnd(helper,0,10);

helper=@(params) -loglikelihood(q_stem,q_leaf,choice_stem,choice_leaf,params(1),params(2),params(3));
A = [];
Aeq = [];
b = [];
Beq = [];
lb = [-Inf 0 0];
ub = [Inf Inf Inf];
[params, lik] = fmincon(helper, [0,1,1], A, b, Aeq, Beq, lb, ub);
%params=fminsearch(helper,[3,0,0]);

bias=params(1);
beta_stem=params(2);
beta_leaf=params(3);

[q_stem_modified,q_leaf_modified,lik]=loglikelihood(q_stem,q_leaf,choice_stem,choice_leaf,bias,beta_stem,beta_leaf);

end