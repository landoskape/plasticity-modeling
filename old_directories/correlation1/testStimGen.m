


NI = 100;
NS = 1;

maxCov = 0.5;
cvRange = sqrt(linspace(maxCov,0,NI));


CV = cvRange' * cvRange;
CV = CV - diag(diag(CV)) + eye(NI);

CH = chol(CV);

T = 100000;
z = randn(T,NI);

data = z*CH;

postCov = cov(data);

figure(1); clf; 
subplot(1,3,1); imagesc(CV);
subplot(1,3,2); imagesc(postCov);
subplot(1,3,3); imagesc(postCov - CV); colorbar


%% -- 

T = 100000; % number samples
NI = 100; % number inputs
NS = 3; % number signals

% Create loading matrix
method = 'slideGauss';
sourceStrength = 3;
autoCorrPrm = 20;
loading = zeros(NS,NI);
switch method
    case 'divide'
        % Give each input one source, distribute evenly
        numInputPerSignal = NI/NS;
        loading = cell2mat(arrayfun(@(signal) arrayfun(@(input) sourceStrength*(input>(signal-1) & input<=signal), (1:NI)/numInputPerSignal, 'uni', 1), (1:NS)','uni', 0));
    case 'divideSoft'
        % Give each input one source, distribute evenly, but with half the
        % inputs on each source at half the source strength
        numInputPerSignal = NI/NS;
        loading = cell2mat(arrayfun(@(signal) arrayfun(@(input) sourceStrength*(input>(signal-1) & input<=signal), (1:NI)/numInputPerSignal, 'uni', 1), (1:NS)','uni', 0));
        idxSoft = rem(floor((0:NI-1)/numInputPerSignal*2),2)==1;
        loading(:,idxSoft) = loading(:,idxSoft)/2;
    case 'slideGauss'
        shiftInputPerSignal = round(NI/(NS));
        firstSignalPeakIdx = round(shiftInputPerSignal/2);
        idxGauss = (1:NI)-NI/2;
        widthGauss = 2/5 * shiftInputPerSignal;
        gaussLoading = exp(-idxGauss.^2/(2*widthGauss^2));
        idxPeakGauss = find(gaussLoading==max(gaussLoading),1);
        gaussLoading = circshift(gaussLoading,firstSignalPeakIdx - idxPeakGauss);
        loading = cell2mat(arrayfun(@(signal) circshift(gaussLoading,(signal-1)*shiftInputPerSignal), (1:NS)', 'uni', 0));
        % Use a gaussian source window (allowing overlap), distribute
end


xinit = randn(T,NI);
s = zscore(filter(ones(1,autoCorrPrm),1,randn(T,NS)),1,1);
varAdjustment = sqrt(sum(loading)+1);

x = xinit./varAdjustment + s*loading./varAdjustment;

stdRates = 5;
meanRate =20;
r = stdRates * x + meanRate;
r(r<0)=0;

[coeff,score] = pca(r);

figure(1); clf; 
subplot(1,3,1); imagesc(r)
subplot(1,3,2); imagesc(cov(r))
subplot(1,3,3); imagesc(coeff)

