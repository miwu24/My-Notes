N = 2.^[1:7];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% This if for f1
a = 0; b = 1;
I_f1 = zeros(1,7);
for i=1:7
    n = N(i);
    I_f1(i) = trapezoid('f1',a,b,n);
end
I_f1;

Err = abs(I_f1-erf(1));
loglog(N,Err)
legend('f1 error')
saveas(gcf,'f1_error.png')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% This is for f2
a = -1; b = 1;
I_f2 = zeros(1,7);
for i=1:7
    n = N(i);
    I_f2(i) = trapezoid('f2',a,b,n);
end
I_f2;

Err = abs(I_f2-1.01);
loglog(N,Err)
legend('f2 error')
saveas(gcf,'f2_error.png')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% This is for f32
a = 0; b = 1;
I_f3 = zeros(1,7);
for i=1:7
    n = N(i);
    I_f3(i) = trapezoid('f3',a,b,n);
end
I_f3;

Err = abs(I_f3-2/3);
loglog(N,Err)
legend('f3 error')
saveas(gcf,'f3_error.png')

