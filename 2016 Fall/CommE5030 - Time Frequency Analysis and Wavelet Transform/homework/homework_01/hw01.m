function hw01()
defsel = input('Do you want to define the variable and input x by yourself? (Y/N)','s');
switch defsel
    case 'Y'
        dt = input('Input the Delta t: ');
        df = input('Input the Delta f: ');
        freq = input('Input the frequency interval: ');
        t1 = input('Input the t2 (Time for signal 1): ');
        t2 = input('Input the t2 (Time for signal 2): ');
        t3 = input('Input the t3 (Time for signal 3): ');
        t = input('What is the samples on t-axis?: ');
        f = input('What is the samples on f-axis?: ');
        x = input('What is the input: ');
        B = input('What is the interval [-B,B] of integration?: ');
    case 'N'
        dt = 0.05;
        df = 0.05;
        freq = [-1500:1500];
        t1 = [0:dt:10-dt];
        t2 = [10:dt:20-dt];
        t3 = [20:dt:30];
        t = [0:dt:30];
        f = [-5:df:5];
        x = [cos(2*pi*t1),cos(6*pi*t2),cos(4*pi*t3)];
        B = 1;
end
tic
y = recSTFT(x,t,f,B);
toc
drawgraph(y,freq,t)
end

function [y] = recSTFT ( x,t,f,B )      % Define recSTFT function
T = length(t);                          % Get the length of vector T
F = length(f);                          % Get the length of Vector F
delta_f = f(2) - f(1);                  % Calculate Delta f
delta_t = t(2) - t(1);                  % Calculate Delta T
Q = B/delta_t;
x = [ x, zeros(1,Q) ];                  % Add zero
N = 1 / (delta_f * delta_t);
X = zeros(T,F);
m = f / delta_f;
n = t / delta_t;
m1 = mod(m,N) + 1;
for i = 1:T
    x1 = zeros(1,floor(N));
    if i <= Q+1
        x1((Q-i+2):(2*Q+1)) = x(1:(Q+i));
    else
        x1(1:(2*Q+1)) = x((i-Q):(i+Q));
    end
    X1 = fft(x1);
    XX1 = zeros(1,length(m));
    XX1 = X1(floor(m1));
    X(i,:) = XX1*delta_t.*exp(1j*2*pi*(Q-(i-1))*m/N);
end
y = X';
end

function drawgraph(y,freq,t)
image(t,freq,abs(y)/max(max(abs(y)))*400)      % Here 400 is a constant can be changed.
colormap(gray(256))                     % Take color in winter set.
set(gca,'Ydir','normal')                % Make the y-axis upsidedown.
set(gca,'Fontsize',12)                  % Change the font size.
xlabel('Time (Sec)','Fontsize',12)      % Define x-axis.
ylabel('Frequency (Hz)','Fontsize',12)  % Define y-axis.
title('STFT of x(t)','Fontsize',12)     % Define the graph title.
end