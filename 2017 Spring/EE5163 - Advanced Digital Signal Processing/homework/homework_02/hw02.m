function hw02()
% H: H(F), the Normalized Frequency Response
% k: The Designed filter h[n] interval [0, N-1]
% N: The points can be calculate with k, N=2k+1
k = input('Input the integer parameter k: ');
filt_Resp_Reso = input('Assign the Transition Band R(m/N): ');
N = 2*k + 1;

% Frequency Sampling Method Filter
% Calculate r, r_1
[fs_filt, F, H_F, r_1, r] = freq_samp(k, @(F) 2*pi()*j*(F-(F>=0.5)), N);

% Calculate the Filter Response "pos" and "Reso"
freq_Resp_pos = (0:(filt_Resp_Reso-1)) * (1/filt_Resp_Reso);
freq_Resp = fft(circshift([fs_filt zeros(1, filt_Resp_Reso-2*k-1)], [0 -k]));

% Draw r[n], h[n], r_1[n]
figure;
% Plot r_1[n]
subplot(311);
stem(0:(N-1), real(r_1));
title('r1[n]');
xlim([-2*k 2*k]);
% Plot r[n]
subplot(312);
stem(-k:k, real(r));
title('r[n]');
xlim([-2*k 2*k]);
% Plot h[n], shift with r[n]
subplot(313);
stem(0:(N-1), real(r));
title('h[n]');
xlim([-2*k 2*k]);

% Draw Impulse / Frequency Response
figure;
% Plot Impulse Response
subplot(211);
stem(-k:k, abs(fs_filt));
title('Impulse Response');
xlabel('time');

% Plot Frequency Response
subplot(212);
plot(...
    F, imag(H_F), 'green o',...
    F, imag(H_F), 'blue',...
    freq_Resp_pos, imag(freq_Resp), 'red'...
);
title('Frequency Response');
xlabel('frequency(Hz)');

function [fs_filt, F, sampleH, r_1, r] = freq_samp(k, H, N)
    % H(F) Sampling
    F = (0:(2*k))/N;
    sampleH = H(F);
    fs_filt = circshift(ifft(sampleH), [0 k]);

    % Calculate r, r_1
    r_1 = ifft(sampleH);
    for i = 1:N
        if i <= k
            r(i) = r_1(i + 1 + k);
        else
            r(i) = r_1(i - k);
        end
    end