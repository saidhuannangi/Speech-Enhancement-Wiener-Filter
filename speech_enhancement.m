clc;
clear;
close all;

%% =========================================================
%% STEP 1: Record Voice
%% =========================================================
fs = 16000;
duration = 5;

recObj = audiorecorder(fs,16,1);
disp('Start speaking...')
recordblocking(recObj,duration);
disp('Recording finished.')

x = getaudiodata(recObj);

if max(abs(x)) > 0
    x = x / max(abs(x));
end

N = length(x);

%% =========================================================
%% STEP 2: Parameters
%% =========================================================
frameLen = 512;
overlap = 256;
hop = frameLen - overlap;
window = hamming(frameLen);

numFrames = floor((N-frameLen)/hop) + 1;

noise_power = ones(frameLen,1)*1e-6;
alpha = 0.95;              % Faster adaptation
threshold = 0.0003;        % More sensitive
beta = 1.5;                % Over-subtraction factor

output = zeros(N,1);

%% =========================================================
%% STEP 3: Enhanced Filtering
%% =========================================================
for i = 1:numFrames
    
    start = (i-1)*hop + 1;
    frame = x(start:start+frameLen-1).*window;
    
    X = fft(frame);
    mag = abs(X);
    phase = angle(X);
    
    power_spectrum = mag.^2;
    
    % Energy-based VAD
    energy = sum(frame.^2);
    
    % Update noise estimate
    if energy < threshold
        noise_power = alpha*noise_power + (1-alpha)*power_spectrum;
    end
    
    % -------- Wiener Gain --------
    H = power_spectrum ./ (power_spectrum + noise_power + eps);
    
    % -------- Spectral Subtraction Boost --------
    clean_mag = mag - beta * sqrt(noise_power);
    clean_mag = max(clean_mag, 0);   % Avoid negative
    
    % Combine both
    enhanced_mag = H .* clean_mag;
    
    % Smooth spectrum (reduce musical noise)
    enhanced_mag = medfilt1(enhanced_mag,5);
    
    % Reconstruct
    X_enhanced = enhanced_mag .* exp(1j*phase);
    enhanced_frame = real(ifft(X_enhanced));
    
    % Overlap-add
    output(start:start+frameLen-1) = ...
        output(start:start+frameLen-1) + enhanced_frame .* window;
end

% Normalize
if max(abs(output)) > 0
    output = output / max(abs(output));
end

%% =========================================================
%% STEP 4: Playback
%% =========================================================
disp('Playing Original...')
sound(x,fs)
pause(duration+1)

disp('Playing Enhanced...')
sound(output,fs)

%% =========================================================
%% STEP 5: Plot
%% =========================================================
t = (0:N-1)/fs;

figure;
subplot(2,1,1)
plot(t,x)
title('Original Speech')
grid on

subplot(2,1,2)
plot(t,output)
title('Enhanced Speech (Improved)')
grid on

%% =========================================================
%% STEP 6: Spectrogram
%% =========================================================
figure;

subplot(2,1,1)
spectrogram(x,512,256,512,fs,'yaxis')
title('Original')

subplot(2,1,2)
spectrogram(output,512,256,512,fs,'yaxis')
title('Enhanced')