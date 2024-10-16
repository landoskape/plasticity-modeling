


%{
Oja's rule on Olshausen & Field images (Or Sanger's Rule)
%}

%% Using Multiple Oja (Or Sanger) Cells
hpath = '/Users/landauland/Dropbox/SabatiniLab/plasticity-modeling/bcm';
dpath = '/Users/landauland/Dropbox/SabatiniLab/plasticity-modeling/sparsenet';

imdata = load(fullfile(dpath,'IMAGES.mat'));
images = imdata.IMAGES;
imSize = size(images,1);
numImages = size(images,3);

% Meta Parameters
imLength = 8;
L = imLength ^ 2;
BUFF = 4;
T = 10000;
numCells = L;
batchSize = 100; 
gamma = 0.001;

method = 'sanger'; % 'oja' or 'sanger'

% Preallocate variables
eta = zeros(numCells,batchSize,T);
u = zeros(numCells,L,T);
u(:,:,1) = rand(numCells,L,1);

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
        ridx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % row idx
        cidx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % column idx
        X(:,bidx)=reshape(images(ridx:ridx+imLength-1,cidx:cidx+imLength-1,imidx),L,1); % batch image
    end
    
    % Compute activity and threshold
    eta(:,:,t) = u(:,:,t) * X;
    switch method
        case 'oja'
            du = gamma * mean(permute(eta(:,:,t),[1 3 2]) .* (permute(X,[3 1 2]) - permute(eta(:,:,t),[1 3 2]) .* u(:,:,t)),3);
            u(:,:,t+1) = u(:,:,t) + du;
        case 'sanger'
            % dw = gamma * (eta * X' - LT[eta * eta'] * u);
            etax = permute(eta(:,:,t),[1 3 2]) .* permute(X,[3 1 2]);
            LT = bsxfun(@times, permute(eta(:,:,t),[1 3 2]), permute(eta(:,:,t),[3 1 2]));
            LT = arrayfun(@(idx) tril(LT(:,:,idx)), permute(1:batchSize,[1 3 2]), 'uni', 0);
            LTu = cell2mat(cellfun(@(lt) lt * u(:,:,t), LT, 'uni', 0));
            du = gamma * mean(etax - LTu, 3);
            u(:,:,t+1) = u(:,:,t) + du;
    end
end

deltau = permute(sqrt(sum(diff(u,1,3).^2,2)),[3 1 2]);

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
        clim = max(abs(u(nc,:,end)));
        xcoord = dispBuffer + (i-1)*(imLength+dispBuffer) + (1:imLength);
        ycoord = dispBuffer + (j-1)*(imLength+dispBuffer) + (1:imLength);
        disparray(xcoord,ycoord) = reshape(u(nc,:,end),imLength,imLength)/clim;
        nc = nc + 1;
    end
end

finalu = u(:,:,end)';




%% 
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.53 0.5 0.45]);

subplot(2,2,[1 3]);
imagesc(disparray)
axis image off
colormap('gray');


subplot(2,2,2); hold on;
meanActivity = permute(mean(eta,2),[3 1 2]);
shadedErrorBar(1:T-1, mean(meanActivity(1:T-1,:),2), std(meanActivity(1:T-1,:),1,2), {'color','k','linewidth',2},0);
%plot(1:T-1, meanActivity, 'color',0.5*[1,1,1],'linewidth',0.5);
%plot(1:T-1, mean(meanActivity,2), 'color','k','linewidth',1.5);

subplot(2,2,4);
shadedErrorBar(1:T-1, mean(deltau,2), std(deltau,1,2), {'color','k','linewidth',2},0);
%plot(1:T-1, cell2mat(deltam), 'color',0.5*[1,1,1],'linewidth',0.5);
%plot(1:T-1, mean(cell2mat(deltam),2), 'color','k','linewidth',1.5);


figure(2); clf;
set(gcf,'units','normalized','outerposition',[0.39 0.2 0.3 0.2]);
imagesc(X' * finalu);



%% Measure Eigenspectrum
eigenBatchsize = 10000;
eigenImages = zeros(L,eigenBatchsize);
for eb = 1:eigenBatchsize
    imidx=ceil(numImages*rand);
    cimage=images(:,:,imidx);
    ridx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % row idx
    cidx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % column idx
    eigenImages(:,eb)=reshape(cimage(ridx:ridx+imLength-1,cidx:cidx+imLength-1),L,1); % batch image
end

[coeff,~,latent] = pca(eigenImages');
beta = coeff' * finalu;

dispCoeff = -ones(dispBuffer + rows*(imLength+dispBuffer),dispBuffer+cols*(imLength+dispBuffer));
nc = 1;
for i = 1:rows
    for j = 1:cols
        clim = max(abs(u(nc,:,end)));
        xcoord = dispBuffer + (i-1)*(imLength+dispBuffer) + (1:imLength);
        ycoord = dispBuffer + (j-1)*(imLength+dispBuffer) + (1:imLength);
        dispCoeff(xcoord,ycoord) = reshape(coeff(:,nc),imLength,imLength)/clim;
        nc = nc + 1;
    end
end

figure(1); clf;
subplot(1,4,[1 2 3]);
imagesc(abs(beta))
caxis(max(abs(beta(:)))*[0,1]);
colormap(hot)

subplot(1,4,4);
plot(mean(beta.^2,2),L:-1:1);




%%

fpath = '/Users/landauland/Dropbox/SabatiniLab/plasticity-modeling/thesisFigures_Modeling/figures';

beta2plot = abs(beta');
beta2plot = beta2plot ./ max(beta2plot,[],2);
com = sum((1:L) .* beta2plot,2) ./ sum(beta2plot,2);
[~,idx] = sort(com,'ascend');

% Beta loading and eigenvalue plot
figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.38 0.2 0.28 0.29]);
set(gca,'units','normalized','position',[0.1800    0.200    0.6825    0.6750]);
shadedErrorBar(1:L, mean(beta2plot,1), std(beta2plot,1,1), {'color','k','linewidth',1.5},1);
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
% print(gcf,'-painters',fullfile(fpath,sprintf('%s_betaDistribution',method)),'-depsc');

% display receptive fields
figure(3); clf;
set(gcf,'units','normalized','outerposition',[0.38 0.49 0.28 0.47]);
imagesc(disparray)
title('Receptive Fields');
axis image off
colormap('gray');
set(gca,'fontsize',24);
% print(gcf,'-painters',fullfile(fpath,sprintf('%s_receptiveFields',method)),'-depsc');


%% -- do the integration metric --

% first compute distances between vectors
NC = size(coeff,2);
normDiff = zeros(NC,NC,NC);
vecDiff = zeros(NC^2,NC);
for nc = 1:NC
    idxNegative = 1*(coeff(:,nc)>=0) + -1*(coeff(:,nc)<0);
    signCorrData = eigenImages .* idxNegative;
    diffVec = permute(signCorrData,[1 3 2]) - permute(signCorrData,[3 1 2]);
    normDiff(:,:,nc) = sqrt(sum(diffVec.^2,3));
    vecDiff(:,nc) = reshape(normDiff(:,:,nc),NC*NC,1);
end
vecSimilarity = exp(-vecDiff/var(vecDiff(:)));

%%
% create a display for vector differences (or similarities)
vecDiffLength = NC;
vecDiffBuffer = imLength*dispBuffer;
dispVecDiff = -ones(vecDiffBuffer + rows*(vecDiffLength+vecDiffBuffer),vecDiffBuffer+cols*(vecDiffLength+vecDiffBuffer));
nc = 1;
for i = 1:rows
    for j = 1:cols
        clim = max(abs(u(nc,:,end)));
        xcoord = vecDiffBuffer + (i-1)*(vecDiffLength+vecDiffBuffer) + (1:vecDiffLength);
        ycoord = vecDiffBuffer + (j-1)*(vecDiffLength+vecDiffBuffer) + (1:vecDiffLength);
        dispVecDiff(xcoord,ycoord) = reshape(vecSimilarity(:,nc),vecDiffLength,vecDiffLength);
        nc = nc + 1;
    end
end

% imagesc(dispVecDiff); colormap('hot'); caxis([0 1]);

%%
netSim = zeros(1,NC); 
for nc = 1:NC
    netSim(nc) = (coeff(:,nc)' * reshape(vecSimilarity(:,nc),NC,NC) * coeff(:,nc)) / sum(coeff(:,nc))^2;
end




