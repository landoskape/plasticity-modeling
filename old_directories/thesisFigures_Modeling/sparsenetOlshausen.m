% sparsenet.m - simulates the sparse coding algorithm
% 
% Before running you must first define A and load IMAGES.
% See the README file for further instructions.

hpath = '/Users/landauland/Dropbox/SabatiniLab/plasticity-modeling/sparsenet';
imData = load(fullfile(hpath,'IMAGES.mat'));
images = imData.IMAGES;
numImages=size(images,3);
imageSize=size(images,1);

T = 4000;
batchSize = 100;
imLength = 8;
L = imLength ^ 2;
BUFF = 4;
numCells = L;

eta = 1.0;
noise_var= 0.01;
beta = 2.2 ;
sigma = 0.316;
tol = 0.01;

% Preallocate
A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

VAR_GOAL=0.1;
S_var=VAR_GOAL*ones(numCells,1);
var_eta=.001;
alpha=.02;
gain=sqrt(sum(A.*A))';

X=zeros(L,batchSize);

% Run Model
msg = '';
for t = 1:T-1
    fprintf(repmat('\b',1,length(msg)));
    msg = sprintf('%d/%d...\n',t,T-1);
    fprintf(msg);
    
    % Get image and create minibatch from subimages
    X = zeros(L,batchSize); 
    for bidx = 1:batchSize % batch image idx
        imidx=ceil(numImages*rand);
        %cimage=images(:,:,imidx);
        ridx=BUFF+ceil((imageSize-imLength-2*BUFF)*rand); % row idx
        cidx=BUFF+ceil((imageSize-imLength-2*BUFF)*rand); % column idx
        X(:,bidx)=reshape(images(ridx:ridx+imLength-1,cidx:cidx+imLength-1,imidx),L,1); % batch image
    end
    
    % calculate coefficients for these data via conjugate gradient routine
    S=cgf_fitS(A,X,noise_var,beta,sigma,tol);
    
    % calculate residual error
    E=X-A*S;
    
    % update bases
    dA=zeros(L,numCells);
    for i=1:batchSize
        dA = dA + E(:,i)*S(:,i)';
    end
    dA = dA/batchSize;
    A = A + eta*dA;
    
    % normalize bases to match desired output variance
    for i=1:batchSize
        S_var = (1-var_eta)*S_var + var_eta*S(:,i).*S(:,i);
    end
    gain = gain .* ((S_var/VAR_GOAL).^alpha);
    normA=sqrt(sum(A.*A));
    for i=1:numCells
        A(:,i)=gain(i)*A(:,i)/normA(i);
    end
end

% Create display array
dispBuffer = 1;
if floor(sqrt(numCells))^2 ~= numCells
    rows = sqrt(numCells/2);
    cols = numCells/rows;
else
    rows = sqrt(numCells);
    cols = rows;
end
disparray = -ones(dispBuffer + rows*(imLength+dispBuffer),dispBuffer+cols*(imLength+dispBuffer));
nc = 1;
for i = 1:rows
    for j = 1:cols
        clim = max(abs(A(:,nc)));
        xcoord = dispBuffer + (i-1)*(imLength+dispBuffer) + (1:imLength);
        ycoord = dispBuffer + (j-1)*(imLength+dispBuffer) + (1:imLength);
        disparray(xcoord,ycoord) = reshape(A(:,nc),imLength,imLength)/clim;
        nc = nc + 1;
    end
end


%% Measure Eigenspectrum
eigenBatchsize = 10000;
eigenImages = zeros(L,eigenBatchsize);
for eb = 1:eigenBatchsize
    imidx=ceil(numImages*rand);
    cimage=images(:,:,imidx);
    ridx=BUFF+ceil((imageSize-imLength-2*BUFF)*rand); % row idx
    cidx=BUFF+ceil((imageSize-imLength-2*BUFF)*rand); % column idx
    eigenImages(:,eb)=reshape(cimage(ridx:ridx+imLength-1,cidx:cidx+imLength-1),L,1); % batch image
end

[coeff,~,latent] = pca(eigenImages');
beta = coeff' * A;


%%
normA = A ./ sqrt(sum(A.*A));

simData = eigenImages'; % load in data here (for flexibility)
similarityMethod = 'cov';
switch similarityMethod
    case 'corr'
        smat = corr(simData);
    case 'cov'
        smat = cov(simData);
end
smatnd = smat - diag(diag(smat)); % arbitrarily remove main diagonal elements
smatnd = abs(smatnd);

normMethod = @(m) m; %m.^2;%abs(m); % flexible method for normalize weight products
S = zeros(1,size(normA,2));
for ii = 1:size(normA,2)
    wprod = normA(:,ii) * normA(:,ii)';
    wprod = wprod - diag(diag(wprod)); % remove diagonal
    wprod = normMethod(wprod);
    S(ii) = sum(wprod .* smatnd, 'all');% / sum(wprod,'all');
end

netvar = sum(latent .* abs(beta),1) ./ sum(abs(beta),1);
sumvar = sum(latent .* abs(beta),1);

P = zeros(1,size(coeff,2));
for ii = 1:size(coeff,2)
    eprod = coeff(:,ii) * coeff(:,ii)';
    eprod = eprod - diag(diag(eprod)); 
    eprod = normMethod(eprod); 
    P(ii) = sum(eprod .* smatnd, 'all');% / sum(eprod,'all');
end

xRange = 0.2;
fColor = 0.3;
xLine = [-0.3 0.3];

figure(1); clf; 
subplot(1,2,1); scatter(netvar, S);
subplot(1,2,2); scatter(latent, P); 

% plot(1 + xRange*rand(size(S))-xRange/2, S, 'linestyle','none','marker','o','markerfacecolor',fColor*[1,1,1],'markerEdgeColor','none');
% plot(2 + xRange*rand(size(P))-xRange/2, P, 'linestyle','none','marker','o','markerfacecolor',fColor*[1,1,1],'markerEdgeColor','none');
% line(xLine+1,mean(S)*[1,1],'color','k','linewidth',2.5);
% line(xLine+2,mean(P)*[1,1],'color','k','linewidth',2.5);
% xlim([0.5 2.5]);

        
%% -- 

y = eigenImages' * coeff; 
xCross = eigenImages * eigenImages';
wCross = permute(coeff,[1 3 2]) .* permute(coeff,[3 1 2]);
vCross = wCross .* xCross / eigenBatchsize;

sameComponent = zeros(1,size(coeff,2));
crossComponent = zeros(1,size(coeff,2));
for cc = 1:size(coeff,2)
    sameComponent(cc) = sum(diag(vCross(:,:,cc)));
    crossComponent(cc) = sum(vCross(:,:,cc),'all') - sameComponent(cc);
end
figure(1); clf;
subplot(1,2,1);
scatter(sameComponent,crossComponent);
subplot(1,2,2);
scatter(latent,crossComponent);


%%
figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.19 0.68 0.7 0.3]);

s1 = subplot(1,4,[1 2]);
imagesc(abs(beta))
caxis(max(abs(beta(:)))*[0,1]);
colormap(s1,'hot')

s3 = subplot(1,4,3);
imagesc(disparray)
axis image off
colormap(s3,'gray');

subplot(1,4,4);
plot(mean(beta.^2,2),L:-1:1);


%% 

fpath = '/Users/landauland/Dropbox/SabatiniLab/plasticity-modeling/thesisFigures_Modeling/figures';

beta2plot = abs(beta');
beta2plot = beta2plot ./ max(beta2plot,[],2);
com = sum((1:L) .* beta2plot,2) ./ sum(beta2plot,2);
[~,idx] = sort(com,'ascend');

% Beta loading image
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.68 0.02 0.28 0.47]);
set(gca,'units','normalized','position',[0.12 0.113 0.7276 0.8120]);
imagesc(beta2plot(idx,:));
xlim([0.5 L+0.5]);
ylim([0.5 numCells+0.5]);
set(gca,'xtick',16:16:64);
set(gca,'ytick',16:16:64);
caxis(max(abs(beta2plot(:)))*[0 1]);
colormap('hot');
set(gca,'ydir','normal');
xlabel('eigenvector');
ylabel('O&F Cell #');
title('Synaptic Weight E-vec \beta''s');
set(gca,'fontsize',24);
% print(gcf,'-painters',fullfile(fpath,'olshausen_weightBetas'),'-depsc');

% Colorbar
figure(11);clf;
set(gcf,'units','normalized','outerposition',[0.68 0.49 0.3 0.47]);
set(gca,'units','normalized','position',[0.12 0.113 0.6 0.8120]);
cb = colorbar();
caxis(max(abs(beta2plot(:)))*[0 1]);
colormap('hot');
ylabel(cb,'\beta','fontsize',24);
set(cb,'ticks',0:0.25:1)
set(gca,'visible','off');
set(gca,'fontsize',24);
% print(gcf,'-painters',fullfile(fpath,'olshausen_colorbar'),'-depsc');

%%
% Beta loading and eigenvalue plot
figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.38 0.2 0.28 0.29]);
set(gca,'units','normalized','position',[0.1800    0.200    0.6825    0.6750]);
shadedErrorBar(1:L, mean(beta2plot,1), std(beta2plot,1,1),'lineprops',{'color','k','linewidth',1.5},'transparent',1);
xlim([0.5 L+0.5]);
ylim([0 1]);
set(gca,'xtick',16:16:64);
set(gca,'ytick',0:1);
xlabel('eigenvector');
ylabel('\beta');
title('eigenvector loading distribution');
set(gca,'fontsize',24);
inset = axes('Position',[0.625 0.6 0.22 0.23]);
plot(1:L, latent, 'color','k','linewidth',1.5);
xlim([0.5 L+0.5]);
set(gca,'xtick',16:16:64);
xlabel('eigenvector');
ylabel('eigenvalue');
box off
set(gca,'fontsize',16);
% print(gcf,'-painters',fullfile(fpath,'olshausen_betaDistribution'),'-depsc');

% display receptive fields
figure(3); clf;
set(gcf,'units','normalized','outerposition',[0.38 0.49 0.28 0.47]);
imagesc(disparray)
title('Receptive Fields');
axis image off
colormap('gray');
set(gca,'fontsize',24);
% print(gcf,'-painters',fullfile(fpath,'olshausen_receptiveFields'),'-depsc');


%% -- plot integration --

% first compute distances between vectors
normDiff = zeros(L,L,L);
cosDiff = zeros(L,L,L);
vecDiff = zeros(L*L,L);
normDiffPC = zeros(L,L,L);
cosDiffPC = zeros(L,L,L);
vecDiffPC = zeros(L*L,L);
for nc = 1:L
    idxNegative = 1*(A(:,nc)>=0) + -1*(A(:,nc)<0);
    signedData = eigenImages .* idxNegative; 
    diffVec = signedData - permute(signedData,[3 2 1]);
    normDiff(:,:,nc) = squeeze(sqrt(sum(diffVec.^2,2)));
    vecDiff(:,nc) = reshape(normDiff(:,:,nc),[],1);
    
    normData = sqrt(sum(signedData.^2,2));
    magScale = normData * normData';
    cosDiff(:,:,nc) = (signedData * signedData') ./ magScale;
    
    % now same for PCs
    idxNegative = 1*(coeff(:,nc)>=0) + -1*(coeff(:,nc)<0);
    signedData = eigenImages .* idxNegative; 
    diffVec = signedData - permute(signedData,[3 2 1]);
    normDiffPC(:,:,nc) = squeeze(sqrt(sum(diffVec.^2,2)));
    vecDiffPC(:,nc) = reshape(normDiffPC(:,:,nc),[],1);
    
    normData = sqrt(sum(signedData.^2,2));
    magScale = normData * normData';
    cosDiffPC(:,:,nc) = (signedData * signedData') ./ magScale;
end
cosSimilarity = (cosDiff+1)/2;
cosSimilarity = cosDiff;
cosSimilarityPC = (cosDiffPC+1)/2;
cosSimilarityPC = cosDiffPC;

%%
temperature = std(normDiff(:));
normSimilarity = exp(-normDiff/temperature);
temperaturePC = std(normDiffPC(:));
normSimilarityPC = exp(-normDiffPC/temperaturePC);
posCoeff = abs(coeff);
netSim = zeros(1,L); 
cosSim = zeros(1,L);
netSimPC = zeros(1,L); 
cosSimPC = zeros(1,L);
for nc = 1:L
    netSim(nc) = (posCoeff(:,nc)' * normSimilarity(:,:,nc) * posCoeff(:,nc)) / sum(posCoeff(:,nc))^2;
    cosSim(nc) = (posCoeff(:,nc)' * cosSimilarity(:,:,nc) * posCoeff(:,nc)) / sum(posCoeff(:,nc))^2;
    
    netSimPC(nc) = (posCoeff(:,nc)' * normSimilarityPC(:,:,nc) * posCoeff(:,nc)) / sum(posCoeff(:,nc))^2;
    cosSimPC(nc) = (posCoeff(:,nc)' * cosSimilarityPC(:,:,nc) * posCoeff(:,nc)) / sum(posCoeff(:,nc))^2;
end
figure(1); clf; 
subplot(1,2,1); hold on;
plot(netvar, cosSim, 'linestyle','none','marker','.','color','k','markersize',24)
plot(latent, cosSimPC, 'linestyle','none','marker','.','color','b','markersize',24)
legend('Sparse Filters','PCs');
subplot(1,2,2); hold on;
plot(netvar, -log2(cosSim), 'linestyle','none','marker','.','color','k','markersize',24)
plot(latent, -log2(cosSimPC), 'linestyle','none','marker','.','color','b','markersize',24)





























