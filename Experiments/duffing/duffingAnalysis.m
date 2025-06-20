clc
close all
clearvars
%% Load Data
data = load("datafiles/500_up.mat");
xMeas = double(data.A);
fIn = double(data.B);
fs = double(1/data.Tinterval);
tEnd = (length(xMeas)-1)*(1/fs);
tStart = 0;
t = tStart:1/fs:tEnd;
t = t';
[S,f,tOut] = stft(xMeas,fs,Window=kaiser(256,5),OverlapLength=220,FFTLength=512);
image(abs(S))
sdb = mag2db(abs(S));
mesh(tOut,f,sdb);
% [t,va,vb] = assemblePicoscopeSignals("datafiles");