%% Test PWLAdapt on the function humps.
fname = 'humps';
xL = 0; xR = 3;
delta = 1/100; hzero = 1/10;
[x,y] = PWLAdapt(fname,xL,xR,delta,hzero);
plot(x,y)
hold on
scatter(x,y,'filled')
title(strcat('\delta= ',num2str(delta),', ',' h_0= ', num2str(hzero)))
hold off
saveas(gcf,'humps.png')