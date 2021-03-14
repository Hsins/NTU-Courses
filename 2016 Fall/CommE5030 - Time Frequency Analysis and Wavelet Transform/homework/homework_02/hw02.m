function hw02()
funsel = input('Function wavread for Older Matlab / audioread for Matlab 2016. (W/A)','s');
switch funsel
    case 'W'
		[a1, fs] = wavread('Chord.wav');		% read audio file
    case 'A'
		[a1, fs] = audioread('Chord.wav');		% read audio file
end
x = a1(:,1);
tau = 0:1/44100:1.6077;
dt = 0.01;
df = 1;
t = 0:dt:max(tau);
f = 20:df:1000;
sgm = 200;
tic												% Timer Start
y = Gabor(x, tau, t, f, sgm);					% Define gabor function
toc												% Timer End
drawgraph(y,f,t)
end

function y = Gabor( x, tau, t, f, sgm )
T = length(tau);
F = length(f);
C = length(t);
dt = t(2) - t(1);
df = f(2) - f(1);
dtau = tau(2) - tau(1);
N = 1/(df * dtau);
B = 1.9143/(sgm^(1/2));							% |a| > 1.9143, the Gaussian Function is small enough to be ignored.
Q = round(B / dtau);
n0 = tau(1) / dtau;
m0 = f(1) / df;
c0 = t(1) / dt;
S = dt / dtau;
y = zeros(C,F);
x1 = zeros(1,N);
window = (sgm^(1/4)).* exp(-sgm.*pi.*((Q-(0:N-1))*dtau).^2);
for n = c0:(c0+C-1)
    win_const = dtau*exp( (1j*2*pi*(Q-n*S)).*(m0:(m0+F-1))./N);
    ns_q = n*S-Q;								% Define variable ns_q = (n*S)-Q
    for q = 0:N-1
        if (q <= (2*Q) && (ns_q+q>=0) && ((ns_q+q+1)<=T))
            x1(q+1) = x(ns_q+q+1);
        else
            x1(q+1) = 0;
        end
    end
    fft_out = fft((window.*x1),N);
    y(n+1-c0,:) = ( win_const .* fft_out(m0+1:m0+F))';
end
y=y';
end

function drawgraph(y,f,t)
image(t,f,abs(y)/max(max(abs(y)))*400)			% Here 400 is a constant can be changed.
colormap(gray(256))								% Take color in gray.
set(gca,'Ydir','normal')						% Make the y-axis upsidedown.
set(gca,'Fontsize',12)							% Change the font size.
xlabel('Time (Sec)','Fontsize',12)				% Define x-axis.
ylabel('Frequency (Hz)','Fontsize',12)			% Define y-axis.
title('Gabor Transform','Fontsize',12)			% Define the graph title.
end