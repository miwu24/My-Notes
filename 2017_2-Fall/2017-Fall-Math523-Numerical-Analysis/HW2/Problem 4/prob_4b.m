
N = [5,11,15];
for i = 1:length(N)
    n = N(i); 
    Nodes_n = arrayfun(@(k) Roots(n+1,k),1:n+1); %% the (n+1) Chebyshev nodes
    y = arrayfun(@(t) abs(t), Nodes_n); %% evalutate f_1d_1 on points in x
    [c] = dividiff(Nodes_n,y); %% the coeffs of Newton form for x,y1
    x_finer = linspace(-1,1,500); %% a finer partition of [-1,1] (for plotting)
    y_finer = arrayfun(@(t) abs(t), x_finer);
    y_inter = arrayfun(@(z) Horner(c, Nodes_n, z), x_finer);
    %%%%%%%%% The following plots the figure

    plot(x_finer,y_finer)
    hold on
    plot(x_finer,y_inter)
    scatter(Nodes_n,y,'fill')
    legend('f(x) = |x|','P_n(x)')
    title(strcat('n=', num2str(n)))
    hold off
    saveas(gcf,strcat('4b_Cheby_',num2str(n),'_pt_interp.png'))
end
