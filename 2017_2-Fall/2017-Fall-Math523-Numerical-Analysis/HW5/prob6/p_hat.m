function [ p_hat_N ] = p_hat( f_N, n )
%P_HAT Summary of this function goes here
%   Detailed explanation goes here
c_hat_N = c_hat(f_N);
N = length(f_N)-1;
phi_N = phi_matrix(N);
p_hat_N = [];
for k=0:N
    p_hat_N = [p_hat_N; dot(c_hat_N(1:n+1), phi_N(1:n+1,k+1))];
end
end

