M = [4,8,16,32,64,128,256]';
x = linspace(0,1,50);

for i=1:length(M)
[a,y] = Problem_7b(M(i));
y_m_err_1 = abs(polyval(flip(a),x)-sin(pi*x));
y_m_err_2 = sqrt((polyval(flip(a),x)-sin(pi*x)).^2);
semilogy(x, y_m_err_1)
hold on
end
legend('m = 4','m = 8', 'm = 16','m = 32','m = 64','m = 128','m = 256')
xlabel('x')
title('1-norm Error Plot')
hold off