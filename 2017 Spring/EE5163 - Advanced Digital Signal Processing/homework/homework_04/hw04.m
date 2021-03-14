function hw04()
    x = input('Input the 1st real signal of length N (row vector): ');
    y = input('Input the 2rd real signal of length N (row vector): ');
    [Fx, Fy] = fftreal(x,y)

function [Fx, Fy] = fftreal(x,y)
    % Make sure that x, y are row vectors
    x = x(:).';
    y = y(:).';

    % Complex Sequence z[n]
    z = x + y*1i;
    Fz = fft(z);

    Fx = (Fz + conj(circshift(fliplr(Fz),[0 1])))/2;        % even sequence
    Fy = (Fz - conj(circshift(fliplr(Fz),[0 1])))/(2*1i);   % odd sequence