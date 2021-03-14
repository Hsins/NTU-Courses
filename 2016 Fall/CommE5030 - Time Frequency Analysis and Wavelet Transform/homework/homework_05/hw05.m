function hw05()
image = double(imread('NTU.jpg'));          % Load Image File
tic                                         % Timer Start
[x1L, x1H1, x1H2, x1H3] = wavedbc8(image);  % Run Function 01
toc                                         % Timer End
figure;
imshow(x1L/255);
figure;
imshow(x1H1);
figure;
imshow(x1H2);
figure;
imshow(x1H3);
tic                                         % Timer Start
x = iwavedbc8(x1L, x1H1, x1H2, x1H3);       % Run Function 02
toc                                         % Timer End
figure;
imshow(x/255);
end

function [x1L, x1H1, x1H2, x1H3] = wavedbc8(x)  % Define Function 01
gn=[-0.0106 0.0329 0.0308 -0.1870 -0.0280 0.6309 0.7148 0.2304];
hn=[0.2304 -0.7148 0.6309 0.0280 -0.1870 -0.0308 0.0329 0.0106];
L = length(gn);
m = size(x, 1);
n = size(x, 2);
d = size(x, 3);
l = zeros(m, n+L-1, d);
h = zeros(m, n+L-1, d);
for i = 1:m
    for j = 1:d
        l(i,:,j) = conv(x(i,:,j), gn);
        h(i,:,j) = conv(x(i,:,j), hn);
    end
end
index = 1:2:size(l, 2);
V1L = l(:,index,:);
V1H = h(:,index,:);
m = size(V1L, 1);
n = size(V1L, 2);
L1 = zeros(m+L-1, n, d);
L2 = zeros(m+L-1, n, d);
H1 = zeros(m+L-1, n, d);
H2 = zeros(m+L-1, n, d);
for i = 1:n
    for j = 1:d
        L1(:,i,j) = conv(V1L(:,i,j), gn);
        L2(:,i,j) = conv(V1L(:,i,j), hn);
        H1(:,i,j) = conv(V1H(:,i,j), gn);
        H2(:,i,j) = conv(V1H(:,i,j), hn);
    end
end
index = 1:2:size(L1,1);
x1L = L1(index,:,:);
x1H1 = L2(index,:,:);
x1H2 = H1(index,:,:);
x1H3 = H2(index,:,:);
end

function x = iwavedbc8(x1L, x1H1, x1H2, x1H3)   % Define Function 02
gn_1 = [0.2304 0.7148 0.6309 -0.0280 -0.1870 0.0308 0.0329 -0.0106];
hn_1 = [0.0106 0.0329 -0.0308 -0.1870 0.0280 0.6309 -0.7148 0.2304];
L = length(hn_1);
m = size(x1L,1);
n = size(x1L,2);
d = size(x1L,3);
index = 1:2:m*2;
RxlL = zeros(m*2, n, d);
Rx1H1 = RxlL;
Rx1H2 = RxlL;
Rx1H3 = RxlL;
RxlL(index,:,:) = x1L;
Rx1H1(index,:,:) = x1H1;
Rx1H2(index,:,:) = x1H2;
Rx1H3(index,:,:) = x1H3;
RL1 = zeros(m*2+L-1, n, d);
RH1 = RL1;
RH2 = RL1;
RH3 = RL1;
for i = 1:n
    for j = 1:d
        RL1(:,i,j)=conv(RxlL(:,i,j), gn_1);
        RH1(:,i,j)=conv(Rx1H1(:,i,j), hn_1);
        RH2(:,i,j)=conv(Rx1H2(:,i,j), gn_1);
        RH3(:,i,j)=conv(Rx1H3(:,i,j), hn_1);
    end
end
v1L = RL1 + RH1;
v1H = RH2 + RH3;
m = size(v1L,1);
n = size(v1L,2);
d = size(v1L,3);
index = 1:2:n*2;
Rv1L = zeros(m, n*2, d);
Rv1H = Rv1L;
Rv1L(:,index,:) = v1L;
Rv1H(:,index,:) = v1H;
xL = zeros(m, n*2+L-1, d);
xH = xL;
for i = 1:m
    for j = 1:d
        xL(i,:,j)=conv(Rv1L(i,:,j), gn_1);
        xH(i,:,j)=conv(Rv1H(i,:,j), hn_1);
    end
end
x = xL + xH;
x(1:L-1,:,:) = [];
x(:,1:L-1,:) = [];
x(size(x,1)-L+1:end,:,:) = [];
x(:,size(x,2)-L+1:end,:) = [];
end