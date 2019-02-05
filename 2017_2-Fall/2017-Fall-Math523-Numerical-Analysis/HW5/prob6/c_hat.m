function [c_hat_N ] = c_hat( f_N )
%C_HAT Summary of this function goes here
%   Detailed explanation goes here
N = length(f_N)-1;
phi_N = phi_matrix(N);
gam_N = gam(N);
c_hat_N = [];
for k=0:N
    c_hat_N = [c_hat_N; 2/(N+1)*dot(f_N,phi_N(k+1,:))/gam_N(k+1)];
end

end

