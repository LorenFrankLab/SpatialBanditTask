function lik=loglikelihood(q_stem,q_leaf,choice_stem,choice_leaf,bias,beta_go,beta_stay,beta_leaf)
% q is a matrix of values of size (# of choices) x (trial number)
% choice are the choices animal make
% assuming animals make choices based on exp(q(t))/sum(exp(q))

[q_stem_modified,q_leaf_modified,lik]=modify_q(q_stem,q_leaf,choice_stem,choice_leaf,bias,beta_go,beta_stay,beta_leaf);

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