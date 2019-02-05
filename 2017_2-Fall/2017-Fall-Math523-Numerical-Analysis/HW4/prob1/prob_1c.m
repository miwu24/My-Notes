T = linspace(0,1,1000);
n = 4; 
for j = 0:4
    n_str = num2str(n); j_str = num2str(j);
    Y = arrayfun(@(t) berstein(n,j,t), T);
    plot(T,Y)
    legend({strcat('$', 'B_{',n_str,'}^{', j_str, '}$')},'Interpreter','latex')
    saveas(gcf,strcat('berstein_',n_str,'_',j_str,'.png'))
end