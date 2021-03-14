function hw01()
N = 21;             % N: Filter Length
k = (N - 1)/2;      % k = (n-1)/2
delta = 0.0001;     % Delta = 0.0001

% Initial
A = zeros(12,12);   % Matrix A
S = zeros(12,1);    % Column Vector S
H = zeros(12,1);    % Column Vector H

% STEP 01
F = [0; 0.05; 0.10; 0.13; 0.16; 0.23; 0.26; 0.29; 0.35; 0.4; 0.45; 0.5];
H = F >= 0.22;
f = 0:delta:0.5;
Hd = f >= 0.2;
W1 = 1.0*(f>=0.22); % Define Weight Function
W2 = 0.5*(f<=0.18);
W = W1+W2;
n = 0;

% Do While Loop
E1 = 99;
E0 = 5;
while((E1-E0 > delta) | (E1-E0 < 0)) 
    % STEP 02
    for i = 1:1:12
        for j = 1:1:11
            A(i,j) = cos(2*pi*(j-1)*F(i));
        end
        A(i,12)=(-1)^(i-1)*1/Wf(F(i));
    end
    S = A^(-1)*H;   % Another Method: S = A\H;
    % STEP 03
    RF = 0;
    for i = 1:11
        RF = RF + S(i)*cos(2*(i-1)*pi*f);
    end
    err = (RF-Hd).*W;
    % STEP 04
    q = 2;
    for i = 2:length(f)-1
        if(err(i) > err(i-1) && err(i) > err(i+1))
            F(q) = delta*i;
            q = q+1;
        end
        if(err(i) < err(i-1) && err(i) < err(i+1))
            F(q) = delta*i;
            q = q+1;
        end
    end
    % STEP 05
    E1 = E0;
    n = n+1;
    [max_value, max_locs] = findpeaks(err);
    [min_value, min_locs] = findpeaks(-err);
    E0 = max(max(abs(err)))
    F = sort([0 (max_locs-1)*delta (min_locs-1)*delta 0.5]);
    F = F';
end

% STEP 06
h(k+1) = S(0+1);
for i = 1:k
    h(k+i+1) = S(i+1)/2;
    h(k-i+1) = S(i+1)/2;
end

% Plot Frequency Response
subplot(211)
plot(f, RF,'k',f ,Hd,'b')
title('Frequency Response');
xlabel('frequency(Hz)');
x = 0:1:20;

% Plot Impulse Response
subplot(212)
stem(x,h)
title('Impulse Response');
xlabel('time');
xlim([-1 21])

function w = Wf(F)
if F >= 0.22
    w = 1;
else
    w = 0;
end
if F <= 0.18
    w = 0.5;
end