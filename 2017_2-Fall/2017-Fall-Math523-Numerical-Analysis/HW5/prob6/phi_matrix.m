function [ phi_N ] = phi_matrix( N )
%PHI_MATRIX computes the matrix [phi_i(x_j)]
%   Detailed explanation goes here
X_N = X(N);
phi_N = ones(N+1,0);
for j = 0:N
phi_N = [phi_N, phi(N,X_N(j+1))];
end

end

