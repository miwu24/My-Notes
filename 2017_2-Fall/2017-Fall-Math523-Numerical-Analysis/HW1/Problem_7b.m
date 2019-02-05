function [a,y] = Problem_7b(m)
v = zeros(m+1,1);
for i = 1:m+1
    v(i) = (i-1)/m;
end
X = flip(vander(v),2);

y = zeros(m+1,1);
for i = 1:m+1
    y(i) = sin(pi*v(i));
end

[a] = Solve(X,y);
end