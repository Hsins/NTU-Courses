function hw03()
% Define Start
t = -9:0.0125:9;
f = -4:0.05:4;
x = exp(j*t.^2/10-j*3*t).*((t>=-9)&(t<=1))+exp(j*t.^2/2+j*6*t).*exp(-(t-4).^2/10);
tic                 % Timer Start
y = wdf(x, t, f);   % Define gabor function
toc                 % Timer End
drawgraph(y,f,t)
end

function y = wdf(x,t,f)
dt = t(2)-t(1);
df = f(2)-f(1);
N = round(1/(2*dt*df));
n1 = round(t(1)/dt);
n2 = round(t(length(t))/dt);
m1 = round(f(1)/df);
m2 = round(f(length(f))/df);
m=mod([m1:m2],N)+1;
Lt = n2-n1+1;
Lf = m2-m1+1;
y = zeros(Lf,Lt);
for n = n1:n2
    U = min(n2-n,n-n1);
    Q = 2*U+1;
    A = x(1-n1+[n-U:n+U]).*(x(1-n1+[n+U:-1:n-U])').';
    A1 = fft(A,N)*2*dt;
    a1 = ceil(Q/N);
    for a2 = 2:a1
        A1 = A1+fft(A((a2-1)*N+1:min(a2*N,Q)),N)*2*dt;
    end
    y(:,n-n1+1)=(A1(m).*exp(j*2*pi/N*U*(m-1))).';
end
end

function drawgraph(y,f,t)
image(t,f,abs(y)/max(max(abs(y)))*400)						% Here 400 is a constant can be changed.
colormap(gray(256))											% Take color in gray.
set(gca,'Ydir','normal')									% Make the y-axis upsidedown.
set(gca,'Fontsize',12)										% Change the font size.
xlabel('Time (Sec)','Fontsize',12)							% Define x-axis.
ylabel('Frequency (Hz)','Fontsize',12)						% Define y-axis.
title('Wigner Distribution Function','Fontsize',12)			% Define the graph title.
end
