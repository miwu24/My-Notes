N = [5,11,15];
for i = 1:length(N)
    n = N(i); 
    x = linspace(-1,1,n+1);
    y1 = arrayfun(@(t) f_1d_1(t), x); %% evalutate f_1d_1 on points in x
    [c_1] = dividiff(x,y1); %% the coeffs of Newton form for x,y1
    x_finer = linspace(-1,1,500); %% a finer partition of [-1,1] (for plotting)
    y1_finer = arrayfun(@(t) f_1d_1(t), x_finer);
    y1_inter = arrayfun(@(z) Horner(c_1, x, z), x_finer);
    %%%%%%%%% The following plots the figure

    plot(x_finer,y1_finer)
    hold on
    plot(x_finer,y1_inter)
    scatter(x,y1,'fill')
    legend('f(x) = e^{2x}sin(3{\pi}x)','P_n(x)')
    title(strcat('n=', num2str(n)))
    hold off
    saveas(gcf,strcat('1_',num2str(n),'_pt_interp.png'))
end

N = [5,11,15];
for i = 1:length(N)
    n = N(i); 
    x = linspace(-1,1,n+1);
    y2 = arrayfun(@(t) f_1d_2(t), x); %% evalutate f_1d_1 on points in x
    [c_2] = dividiff(x,y2); %% the coeffs of Newton form for x,y1
    x_finer = linspace(-1,1,500); %% a finer partition of [-1,1] (for plotting)
    y2_finer = arrayfun(@(t) f_1d_2(t), x_finer);
    y2_inter = arrayfun(@(z) Horner(c_2, x, z), x_finer);
    %%%%%%%%% The following plots the figure

    plot(x_finer,y2_finer)
    hold on
    plot(x_finer,y2_inter)
    scatter(x,y2,'fill')
    legend('f(x) = |x|','P_n(x)')
    title(strcat('n=', num2str(n)))
    hold off
    saveas(gcf,strcat('2_',num2str(n),'_pt_interp.png'))
end
