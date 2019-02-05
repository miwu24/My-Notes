x = [0.6;0.7;0.9];
y = arrayfun(@(t) f(t),x);
[c] = dividiff(y,x);
[x_star] = Horner(c, y, 0);
x_star %% This is the estimated zero