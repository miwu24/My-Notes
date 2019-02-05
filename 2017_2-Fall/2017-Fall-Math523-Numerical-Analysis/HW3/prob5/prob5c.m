a = 0; b = 1;N = 11;
x = linspace(a,b,N)';
y = arrayfun(@(t) t^(5/2),x);
scatter(x,y, 'filled')
hold on
plot(x,y)
[c] = not_a_knot(a,b,N,y);
for i = 1:N-1
    x_i = linspace((i-1)/10, i/10, 11)';
    y_p_i = polyval(flip(c(i, :)), x_i-x_i(1));
    plot(x_i, y_p_i)
end
print -r1500
saveas(gcf,'CubicSpline.tiff')
hold off