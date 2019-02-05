prob_3a;

x = [0.6;0.7;0.8;0.9];
y = arrayfun(@(t) f(t),x);
[c] = dividiff(y,x);
[x_star_2] = Horner(c, y, 0);
x_star_2 %% This is the estimated zero
est_true_err = abs(x_star-x_star_2)