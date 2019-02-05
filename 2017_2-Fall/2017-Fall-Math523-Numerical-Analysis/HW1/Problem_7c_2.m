M = [4,8,16,32,64,128,256]';
OneNorm = [];
TwoNorm = [];
for i=1:length(M)
[OneNormErr,TwoNormErr] = Problem_7c(M(i));
OneNorm = [OneNorm;OneNormErr];
TwoNorm = [TwoNorm;TwoNormErr];
end

semilogy(M,OneNorm)
hold on
semilogy(M,TwoNorm)
legend('1-norm error','2-norm error')
xlabel('m')
hold off