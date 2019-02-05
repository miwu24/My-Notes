%% Test PWLAdapt on the function sqrt.
fname = 'sqrt';
xL = 0; xR = 1;
delta = 1/1000; hzero = 1/100;
[x,y] = PWLAdapt(fname,xL,xR,delta,hzero);
plot(x,y)
hold on
scatter(x,y,'filled')
title(strcat('\delta= ',num2str(delta),', ',' h_0= ', num2str(hzero)))
hold off
saveas(gcf,'sqrt.png')