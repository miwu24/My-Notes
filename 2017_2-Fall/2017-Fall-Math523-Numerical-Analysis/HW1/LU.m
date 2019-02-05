function [P,L,U] = LU(A)
P = eye(size(A,1));
L = eye(size(A,1));
U = A;
m = size(A,1);
for k=1:m-1
    [Max,i] = max(abs(U(k:m,k)));
    i = i+k-1;
    U_k = U(k,k:m); U_i = U(i,k:m);
    U(k,k:m) = U_i; 
    U(i,k:m) = U_k;
    L_k = L(k,1:k-1); L_i = L(i,1:k-1);
    L(k,1:k-1) = L_i;
    L(i,1:k-1) = L_k;
    P_k = P(k,:); P_i = P(i,:);
    P(k,:) = P_i; P(i,:) = P_k;
    for j = k+1:m
        L(j,k) = U(j,k)/U(k,k);
        U(j,k:m) = U(j,k:m)-L(j,k)*U(k,k:m);
    end
end
end
    
