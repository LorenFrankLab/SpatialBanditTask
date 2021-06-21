function lik=loglikelihood(q_stem,q_leaf,choice_stem,choice_leaf,bias,beta_stem,beta_leaf)
% q is a matrix of values of size (# of choices) x (trial number)
% choice are the choices animal make
% assuming animals make choices based on exp(q(t))/sum(exp(q))
lik=0;
if nargin<3
    bias=0;
end

prev_stem=NaN;
for t=1:length(choice_stem)
    if t>1
        q_=beta_stem.*q_stem(:,t);
        q_(choice_stem(t-1))=q_(choice_stem(t-1))+bias;
    else
        q_=q_stem(:,1);
    end

    lik=lik+q_(choice_stem(t))-logsumexp(q_);
    if (prev_stem ~= choice_stem(t))
        %log likelihood of leaf choice on switches only
        lik = lik + beta_leaf * q_leaf(choice_leaf(t),t) - logsumexp(beta_leaf * q_leaf(:,t));
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