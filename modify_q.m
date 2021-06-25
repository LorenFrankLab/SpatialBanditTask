function [q_stem_modified,q_leaf_modified,lik]=modify_q(q_stem,q_leaf,choice_stem,choice_leaf,bias,beta_go,beta_stay,beta_leaf)
% q is a matrix of values of size (# of choices) x (trial number)
% choice are the choices animal make
% assuming animals make choices based on exp(q(t))/sum(exp(q))
lik=0;
if nargin<3
    bias=0;
end

prev_stem=NaN;
q_stem_modified=q_stem;
q_leaf_modified=q_leaf;
for t=1:length(choice_stem)
    q_=beta_go.*q_stem(:,t);
    if t>1
        other_leaf=2-floor((choice_leaf(t))/(2*choice_stem(t)))+(choice_stem(t)-1)*2;
        q_(choice_stem(t))=beta_stay.*q_leaf(other_leaf,t)+bias;
    end
    
    q_stem_modified(:,t)=q_;
    q_leaf_modified(:,t)=beta_leaf * q_leaf(:,t);

    lik=lik+q_(choice_stem(t))-logsumexp(q_);
    if and(t>1,prev_stem ~= choice_stem(t))
        %log likelihood of leaf choice on switches only
        lik = lik + q_leaf_modified(choice_leaf(t),t) - logsumexp(beta_leaf * q_leaf_modified(:,t));
    end
    
    prev_stem=choice_stem(t);
end

end

function s = logsumexp(a, dim)
% Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
% Default is dim = 1 (columns).
% logsumexp(a, 2) will sum across rows instead of columns.
% Unlike matlab's "sum", it will not switch the summing direction
% if you provide a row vector.
% Written by Tom Minka
% (c) Microsoft Corporation. All rights reserved.

a=reshape(a,length(a),1);
if nargin < 2
  dim = 1;
end

% subtract the largest in each column
[y, i] = max(a,[],dim);
dims = ones(1,ndims(a));
dims(dim) = size(a,dim);
a = a - repmat(y, dims);
s = y + log(sum(exp(a),dim));
i = find(~isfinite(y));
if ~isempty(i)
  s(i) = y(i);
end
end