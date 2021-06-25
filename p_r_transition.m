function [p,a,b]=p_r_transition(x,mean,width,dr,r)

    p=zeros(size(x));
    a=1/width*mean;
    b=1/width-a;
    if and(a>0,b>0)
        p=betapdf(x,a,b);
        p_all=betapdf(r,a,b);
        p_sum=sum(p_all(~isinf(p_all))*dr);

        %*dr/sum(betapdf(r,a,b)*dr);
        inf_index=isinf(p);
        if sum(inf_index)>0
            p(inf_index)=0;
        end
        p=p*dr/p_sum;
    end
end