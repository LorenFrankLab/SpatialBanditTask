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