function hw03()
    score = [1,1,2,3,4,5,5,5,5,4,3,5,5,5,5,6,7,8,3,3,6,5,3,2,1,3,6,6,5,5,3,2,3,4,5,3,2];
    beat =  [1,0.5,0.5,0.5,0.5,0.5,1,0.5,1,0.5,1,1,0.5,1,1,1,1,2,0.5,1,0.5,2.5,0.5,0.5,1.5,0.5,1,1,1,1.5,1,0.5,0.5,0.5,0.5,0.5,2];
    name = input('File name: ');
    key = input('Determine the key (C: 1, D: 2, E: 3, ..., B: 7): ');
    fs = input('Determine the fs: ');
    dt = input('Determine the beat Unit: ');
    melody = GenMelody(score, beat, key, fs, dt);
    sound(melody, fs);
    audiowrite(strcat(name, '.wav'), melody, fs);
              sound(melody, fs);

function melody = GenMelody(score, beat, key, fs, dt)
    % 表列了三個八度的 Do ~ Si 的半音標誌 p
    % MIDI 標準中: p = 69 + 12*log(2,f/f0)
    %             f = f0 * 2^((p-69)/12)
    p = [60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86,88,89,91,93,95,96];
    % 根據 MIDI 標準換算所得到的各調性基礎頻率 f0
    freq = [440.00,493.88,523.25,587.31,659.26,739.99,830.61];
    melody=[];
    for i = 1:length(score);
        time = linspace(0, beat(i)*dt, fs*beat(i)*dt);
        f = freq(key)*2^((p(score(i))-69)/12);
        y = sin(2*pi*f*time);
        melody=[melody y];
    end