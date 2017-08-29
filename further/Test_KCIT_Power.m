% This test is used to compare the PCIT and the KCIT

% Load Wine data set and store as Wine
% from https://archive.ics.uci.edu/ml/datasets/wine

rng(1)

n = size(Wine,1)

X1 = table2array(Wine(:,2))
X2 = table2array(Wine(randperm(n),3))
noise = table2array(Wine(randperm(n),6))

n_range = [100,200,500,1000,2000,5000]
B = 200
idx = 1
power = ones(6,1)
time = ones(6,1)
for sample_size = n_range
    mistakes = 0;
    tic
    for i = 1:B
        X1_round = X1(randsample(n,sample_size, true));
        X2_round = X2(randsample(n,sample_size,true));
        noise_round = ((rand(sample_size,1) > 0.5) * 2 - 1) .* sqrt(noise(randsample(n,sample_size,true)));
        Z = log(X1_round).*exp(X2_round) + noise_round;

        found = indtest_new(X1_round,X2_round,Z,[]) > 0.05;
        mistakes = mistakes + found;
    end
    power(idx) = 1 - mistakes / B;
    time(i) = toc;
    idx = idx + 1
end