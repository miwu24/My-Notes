%% This box plots T_0 and save it as png file
n = 0;
x = linspace(-1,1,100);
plot(x,Chebyshev(n,x)) %% plot T_n
title(strcat('n=',num2str(n)))
saveas(gcf,strcat('4a_Cheby_',num2str(n),'.png'))

%% This box plots T_n for n=1:5, including their roots and extrema, resp., and 
%% save them as png files.
N = [1:5];
x = linspace(-1,1,100);
for i=2:length(N)
    n = N(i);
    plot(x,Chebyshev(n,x)) %% plot T_n
    hold on
    Roots_n = arrayfun(@(k) Roots(n,k),1:n); %% compute the roots
    Ext_n = arrayfun(@(k) Extrema(n,k),0:n); %% compute the extrema
    scatter(Roots_n, zeros(n,1),'filled')
    scatter(Ext_n, Chebyshev(n, Ext_n),'filled')
    title(strcat('n=',num2str(N(i))))
    saveas(gcf,strcat('4a_Cheby_',num2str(N(i)),'.png'))
    hold off
end