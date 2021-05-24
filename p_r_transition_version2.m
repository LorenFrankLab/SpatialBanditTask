function p=p_r_transition_version2(x,prev_r,v,dr,r)
number=length(r);
p=zeros(length(x),1);
for i=1:length(x)
    if and(x(i)<prev_r+dr,x(i)>=prev_r)
        p(i)=(1-v);
    else
        p(i)=v/(number-1);
    end
end
    
end