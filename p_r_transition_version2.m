function p=p_r_transition_version2(x,prev_r_index,v,r)
number=length(r);
p=zeros(length(x),1);

for i=1:length(x)
    if x(i)==prev_r_index
        p(i)=(1-v);
    else
        p(i)=v/(number-1);
    end
end
    
end