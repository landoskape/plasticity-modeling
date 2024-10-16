% test something about eigenvectors and expected value of correlations

N = 5;
evec = eye(N);
eval = 2.^-(1:N);

T = 100000;
kappa = sqrt(eval).*randn(T,N);

data = kappa * pinv(evec);

[coeff,score,latent] = pca(data);

figure(1); clf;
subplot(1,2,1); imagesc(eval); 
subplot(1,2,2); imagesc(coeff);


%% -- orthonormal with gram schmidt --

A = [2 -3 -1; 1 1 -1; 0 1 -1];
B = orth(A);
N = length(A);
evec = B;
eval = 2.^-(1:N);
T = 10000;
kappa = sqrt(eval).*randn(T,N);
data = kappa * evec';
[coeff,score,latent] = pca(data);
for n = 1:N
    if sign(coeff(1,n))~=sign(evec(1,n))
        coeff(:,n) = -coeff(:,n);
        score(:,n) = -score(:,n);
    end
end
% figure(1); clf;
% subplot(1,2,1); imagesc(B); 
% subplot(1,2,2); imagesc(coeff);
eval .* evec;
data' * kappa / T;

w = randn(3,1);
w = w / norm(w);
y = data * w;

tRun = 10000;
wrun = zeros(3,tRun);
wrun(:,1) = w;
for t = 2:tRun
    y = data * wrun(:,t-1);
    p = y' * data / T;
    wrun(:,t) = wrun(:,t-1) + 0.1*p';
    wrun(:,t) = wrun(:,t) / norm(wrun(:,t));
end

clc
kw = w' * evec;
pi = y' * data / T;
pMatrix = (evec.*eval).*(w'*evec);
%sum(evec.^2 .* eval.*2,2)'
%p.^2




yrun = data * wrun(:,end);
prun = yrun' * data / T;
fprintf('pRandom: %.2f, pRun: %.2f\n',sum(w.*pi'), sum(wrun(:,end).*prun'));

  

%% -- do it again with noiser inputs and more noise --

N = 3;
A = [2 -3 -1; 1 1 -1; 0 1 -1];
B = orth(A);
evec = B;
eval = 2.^-(1:N);
T = 100000;
kappa = sqrt(eval).*randn(T,N);

smoothScores = 10;
kappa = filter(ones(1,smoothScores)/sqrt(smoothScores),1,kappa,[],1);
data = kappa * evec';
[coeff,score,latent] = pca(data);
for n = 1:N
    if sign(coeff(1,n))~=sign(evec(1,n))
        coeff(:,n) = -coeff(:,n);
        score(:,n) = -score(:,n);
    end
end

figure(1); clf;
subplot(1,2,1); imagesc(coeff); 
subplot(1,2,2); imagesc(evec);

w = randn(N,1);
w = w / norm(w);
y = data * w;
pi = y' * data / T;
kw = w' * evec; 
ptheory = (evec * (eval' .* kw'))';

%
tRun = 10000;
wrun = zeros(N,tRun);
wrun(:,1) = w;
%wrs = zeros(N,tRun);
%wrs(:,1) = w;
for t = 2:tRun
    y = data * wrun(:,t-1);
    p = y' * data / T;
    wrun(:,t) = wrun(:,t-1) + 0.1*p';
    wrun(:,t) = wrun(:,t) / norm(wrun(:,t));
end

clc
kw = w' * evec;
y = data * w;
pi = y' * data / T;

pMatrix = (evec.*eval).*kw;
% sum(evec.^2 .* eval.*2,2)'
%p.^2

nw = @(N) ones(N,1)/norm(ones(N,1));
w1 = nw(N);
d1 = repmat(data(:,1),1,N);
y1 = d1 * w1;
p1 = y1' * d1 / T;

yrun = data * wrun(:,end);
prun = yrun' * data / T;
% yrs = spikes * wrs(:,end);
% prs = yrs' * spikes / T;

kwrun = wrun(:,end)' * evec;
pthrun = (evec * (eval' .* kwrun'))';

kw = evec(:,1)' * evec; 
ptheory = (evec * (eval' .* kw'))';

fprintf('pRandom: %.2f, pRun: %.2f, pTheory: %.2f, p1: %.2f\n',sum(w.*pi'), sum(wrun(:,end).*prun'), sum(ptheory*evec(:,1)), p1*w1);
%fprintf('pRandom: %.2f, pRun: %.2f, pSpikes: %.2f\n',sum(w.*pi'), sum(wrun(:,end).*prun'), sum(wrs(:,end).*prs'));



%% -- 
N = 11;
A = magic(N);
evec = orth(A);
eval = 1.3.^-(1:N);
T = 10000;
kappa = sqrt(eval).*randn(T,N);
data = kappa * evec';

w = zeros(N,T);
w(:,1) = randn(N,1);
w(:,1) = w(:,1)/norm(w(:,1));

dt = 0.1;
for t = 2:T
    cy = data(t,:) * w(:,t-1);
    dw = cy * (data(t,:)' - cy*w(:,t-1));
    w(:,t) = w(:,t-1) + dt * dw;
end


