N = 2^12;
hold off
X_N = linspace(0,2*pi,N+1);
f_N = arrayfun(@(x) fun_prob5(x),X_N);
DF_f_N = abs(fft(f_N));
X_plot = [1:N];
plot(X_plot, log(DF_f_N(X_plot))-log(1./(X_plot.^3)))
legend(strcat('N = ',num2str(N)))
title('log(fft)-log(1/k^3)')
saveas(gcf,strcat(num2str(N),'.png'))
