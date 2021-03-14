function hw04()
t = [0: 0.01: 10];
x = 0.2*t + cos(2*pi*t) + 0.4*cos(10*pi*t);
thr = 0.2;
tic													% Timer Start
y = hht(x, t, thr);
toc													% Timer End
drawgraph(y, t)
end

function y = hht(x, t, thr)
n = 1;
dt = t(2) - t(1);
while (1)
    if (length(findpeaks(x)) <= 3)
        y(n,:) = x;
        break
    end
    
    temp = x;
    test = 1;
    k = 1;
    while (test == 1 && k < 3)
        test = 0;
        [max maxloc] = findpeaks(temp);         	% Step02
        peaks = spline((maxloc-1)*dt, max, t);    	% Step03
        [neg_min minloc] = findpeaks(temp*(-1));	% Step04
        min = (-1)*(neg_min);
        dips = spline((minloc-1)*dt, min, t);     	% Step05
        z = (peaks + dips) / 2;                    	% Step06_1
        h = (temp - z);								% Step06_2
        
        % Step07
        hpeaks = findpeaks(h);
        hdips = (-1)*findpeaks(-1*h);
        for i = 1:(length(hpeaks)-1)
            if ((hpeaks(i) <= 0) || (hdips(i) >= 0) || (abs((hpeaks(i) + hdips(i))/2) >= thr))
                temp = h;
                test = 1;
                break
            end
        end
        k = k + 1;
    end
    y (n,:) = h;
    % Step08
    x = x - h;
    n = n + 1;
end

end

function drawgraph(y, t)
subplot(3,1,1);				% Draw IMF01
plot(t,y(1,:));
title('IMF1');
subplot(3,1,2);				% Draw IMF02
plot(t,y(2,:));
title('IMF2');
subplot(3,1,3);				% Draw the Trend of Signal
plot(t,y(3,:));
title('Trend');
end