N = 10;
phi_N = phi_matrix(N);
X_N = X(N);
%%%%%% This block is for the function e^(-x).
f_N = arrayfun(@(x) exp(-x),X_N);
c_hat_N = c_hat(f_N);
e_N_2 = zeros(N+1,1);
e_N_inf = zeros(N+1,1);
for n=0:N
    e_N_2(n+1) = sqrt(2/(N+1))*norm(p_hat( f_N, n )-f_N);
    e_N_inf(n+1) = max(abs(p_hat( f_N, n )-f_N));
end
[c_hat_N e_N_2 e_N_inf]
figure(1)
plot(X_N, p_hat( f_N, 2))
hold on
plot(X_N, p_hat( f_N, 4))
plot(X_N, p_hat( f_N, 10))
X_plot = linspace(-1,1,1000);
plot(X_plot,arrayfun(@(x) exp(-x),X_plot))
legend('p_2', 'p_4', 'p_{10}', 'f=e^{-x}')
saveas(gcf,'plot_1.png')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% this block is for the function log(x+2)
f_N = arrayfun(@(x) log(x+2),X_N);
c_hat_N = c_hat(f_N);
e_N_2 = zeros(N+1,1);
e_N_inf = zeros(N+1,1);
for n=0:N
    e_N_2(n+1) = sqrt(2/(N+1))*norm(p_hat( f_N, n )-f_N);
    e_N_inf(n+1) = max(abs(p_hat( f_N, n )-f_N));
end
[c_hat_N e_N_2 e_N_inf]
figure(2)
plot(X_N, p_hat( f_N, 2))
hold on
plot(X_N, p_hat( f_N, 4))
plot(X_N, p_hat( f_N, 10))
X_plot = linspace(-1,1,1000);
plot(X_plot,arrayfun(@(x) log(x+2),X_plot))
legend('p_2', 'p_4', 'p_{10}', 'f=log(x+2)')
saveas(gcf,'plot_2.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% this block is for the function sqrt(1+x)
f_N = arrayfun(@(x) sqrt(1+x),X_N);
c_hat_N = c_hat(f_N);
e_N_2 = zeros(N+1,1);
e_N_inf = zeros(N+1,1);
for n=0:N
    e_N_2(n+1) = sqrt(2/(N+1))*norm(p_hat( f_N, n )-f_N);
    e_N_inf(n+1) = max(abs(p_hat( f_N, n )-f_N));
end
[c_hat_N e_N_2 e_N_inf]
figure(3)
plot(X_N, p_hat( f_N, 2))
hold on
plot(X_N, p_hat( f_N, 4))
plot(X_N, p_hat( f_N, 10))
X_plot = linspace(-1,1,1000);
plot(X_plot,arrayfun(@(x) sqrt(1+x),X_plot))
legend('p_2', 'p_4', 'p_{10}', 'f=sqrt(1+x)')
saveas(gcf,'plot_3.png')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% this block is for the function abs(x)
f_N = arrayfun(@(x) abs(x),X_N);
c_hat_N = c_hat(f_N);
e_N_2 = zeros(N+1,1);
e_N_inf = zeros(N+1,1);
for n=0:N
    e_N_2(n+1) = sqrt(2/(N+1))*norm(p_hat( f_N, n )-f_N);
    e_N_inf(n+1) = max(abs(p_hat( f_N, n )-f_N));
end
[c_hat_N e_N_2 e_N_inf]
figure(4)
plot(X_N, p_hat( f_N, 2))
hold on
plot(X_N, p_hat( f_N, 4))
plot(X_N, p_hat( f_N, 10))
X_plot = linspace(-1,1,1000);
plot(X_plot,arrayfun(@(x) abs(x),X_plot))
legend('p_2', 'p_4', 'p_{10}', 'f = abs(x)')
saveas(gcf,'plot_4.png')


