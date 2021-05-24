function [x2,inBetween]=plot_std(y,std_dev)
x=1:length(y);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];