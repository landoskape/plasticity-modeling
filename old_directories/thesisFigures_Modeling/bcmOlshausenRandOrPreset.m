
% Using Multiple Cells
hpath = '/Users/landauland/Dropbox/SabatiniLab/plasticity-modeling/bcm';
dpath = '/Users/landauland/Dropbox/SabatiniLab/plasticity-modeling/sparsenet';

imdata = load(fullfile(dpath,'IMAGES.mat'));
images = imdata.IMAGES;
imSize = size(images,1);
numImages = size(images,3);

imLength = 8;
L = imLength ^ 2;
BUFF = 4;
T = 5000;
numCells = L;

%avgMethod = 'batch'; % 'batch' or 'exponential'
batchSize = 20; 
imMethod = 'randSelect'; % 'randSelect' or 'preset'

phi = @(c, theta) c .* (c - theta); % Plasticity activation function
c0 = 1;
eta = 0.01; % learning rate
eps = 0.001; % decay rate

% Preallocate variables
c = cell(1,numCells);
chat = cell(1,numCells);
m = cell(1,numCells);
for nc = 1:numCells
    c{nc} = zeros(T,batchSize); % activity vector
    chat{nc} = zeros(T,1); % average squared activity
    m{nc} = zeros(T,L); % weight vector
    m{nc}(1,:) = rand(1,L); % initialize
end

if strcmp(imMethod,'preset')
    X = zeros(L,batchSize); 
    for bidx = 1:batchSize % batch image idx
        imidx=ceil(numImages*rand);
        cimage=images(:,:,imidx);
        ridx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % row idx
        cidx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % column idx
        X(:,bidx)=reshape(cimage(ridx:ridx+imLength-1,cidx:cidx+imLength-1),L,1); % batch image
    end
    X = X./sqrt(sum(X.^2,1));
    %X = zscore(X,1,1);
end

% Run Model
msg = '';
for t = 1:T-1
    fprintf(repmat('\b',1,length(msg)));
    msg = sprintf('%d/%d...\n',t,T-1);
    fprintf(msg);
    
    % Image Library
    switch imMethod
        case 'randSelect'
            % Get image and create minibatch from subimages
            imidx=ceil(numImages*rand);
            cimage=images(:,:,imidx);
            X = zeros(L,batchSize); 
            for bidx = 1:batchSize % batch image idx
                ridx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % row idx
                cidx=BUFF+ceil((imSize-imLength-2*BUFF)*rand); % column idx
                X(:,bidx)=reshape(cimage(ridx:ridx+imLength-1,cidx:cidx+imLength-1),L,1); % batch image
            end
        case 'preset'
            % use preset X, already constructed!
    end
    
    % Compute activity and threshold
    for nc = 1:numCells
        c{nc}(t,:) = m{nc}(t,:) * X;
        chat{nc}(t) = mean((c{nc}(t,:)/c0).^2);
        dm = mean(phi(c{nc}(t,:), chat{nc}(t)) .* X,2)' - eps * m{nc}(t,:);
        m{nc}(t+1,:) = m{nc}(t,:) + eta * dm;
    end
end
deltam = cell(1,numCells);
for nc = 1:numCells
    deltam{nc} = sqrt(sum(diff(m{nc},1,1).^2,2));
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
        clim = max(abs(m{nc}(end,:)));
        xcoord = dispBuffer + (i-1)*(imLength+dispBuffer) + (1:imLength);
        ycoord = dispBuffer + (j-1)*(imLength+dispBuffer) + (1:imLength);
        disparray(xcoord,ycoord) = reshape(m{nc}(end,:),imLength,imLength)/clim;
        nc = nc + 1;
    end
end

finalm = cell2mat(cellfun(@(c) c(end,:)', m, 'uni', 0));

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
beta = coeff' * finalm;

figure(1); clf;
subplot(1,4,[1 2 3]);
imagesc(abs(beta))
caxis(max(abs(beta(:)))*[0,1]);
colormap('hot')

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
inset = axes('Position',[0.25 0.65 0.22 0.18]);
plot(1:L, latent, 'color','k','linewidth',1.5);
xlim([0.5 L+0.5]);
set(gca,'xtick',16:16:64);
xlabel('eigenvector');
ylabel('eigenvalue');
box off
set(gca,'fontsize',16);
print(gcf,'-painters',fullfile(fpath,'bcm_betaDistribution'),'-depsc');

% display receptive fields
figure(3); clf;
set(gcf,'units','normalized','outerposition',[0.38 0.49 0.28 0.47]);
imagesc(disparray)
title('Receptive Fields');
axis image off
colormap('gray');
set(gca,'fontsize',24);
print(gcf,'-painters',fullfile(fpath,'bcm_receptiveFields'),'-depsc');







%% 
%{
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.53 0.5 0.45]);

subplot(3,2,[1 3 5]);
imagesc(disparray)
axis image off
colormap('gray');


subplot(3,2,2); hold on;
meanActivity = cell2mat(cellfun(@(c) mean(c,2), c, 'uni', 0));
shadedErrorBar(1:T-1, mean(meanActivity(1:T-1,:),2), std(meanActivity(1:T-1,:),1,2), {'color','k','linewidth',2},0);
%plot(1:T-1, meanActivity, 'color',0.5*[1,1,1],'linewidth',0.5);
%plot(1:T-1, mean(meanActivity,2), 'color','k','linewidth',1.5);

subplot(3,2,4);
thing2plot = 'selectivity'; % 'chat' or 'selectivity'
switch thing2plot
    case 'chat'
        meanCHat = cell2mat(chat);
        shadedErrorBar(1:T-1, mean(meanCHat(1:T-1,:),2), std(meanCHat(1:T-1,:),1,2), {'color','k','linewidth',2},0);
        %plot(1:T-1, meanCHat(1:T-1,:), 'color',0.5*[1,1,1],'linewidth',0.5);
        %plot(1:T-1, mean(meanCHat(1:T-1,:),2), 'color','k','linewidth',1.5);
    case 'selectivity'
        allc = cat(3, c{:});
        allc = abs(allc); % use some transform
        selectivity = squeeze(mean(allc,2)./max(allc,[],2));
        shadedErrorBar(1:T, mean(selectivity,2), std(selectivity,1,2), {'color','k','linewidth',2},0);
end

subplot(3,2,6);
shadedErrorBar(1:T-1, mean(cell2mat(deltam),2), std(cell2mat(deltam),1,2), {'color','k','linewidth',2},0);
%plot(1:T-1, cell2mat(deltam), 'color',0.5*[1,1,1],'linewidth',0.5);
%plot(1:T-1, mean(cell2mat(deltam),2), 'color','k','linewidth',1.5);


figure(2); clf;
set(gcf,'units','normalized','outerposition',[0.39 0.2 0.3 0.2]);
imagesc(X' * finalm);
%}



