function [iaf,spikes,y,trackWeights] = runo2_iafCM(runIdx,sDepIdx,sCorIdx)
rng('shuffle');

sDepArray = [1.05, 1.025];
sCorArray = linspace(0.2, 0, 10);
silentDepression = sDepArray(sDepIdx);
silentCorrelation = sCorArray(sCorIdx);

T = 5000*1000;

options.dt = 0.001;
options.maxWeight = 0.015; % 150pS
options.inhWeight = 0.05; % 500pS
options.propNecessaryNMDA = inf;
options.loseSynapseRatio = -inf;
options.newSynapseRatio = 0.1;
options.numActive = 40;
options.numSilent = 0;
options.numSynapses = 25;
options.conductanceThreshold = 0;% 0.2;
options.plasticityRate = [0.005, 0.005];
options.depValue = [1.05, silentDepression];
options.potentiationTau = 0.02;
options.depressionTau = 0.02;
options.homeostatisLimit = 4.2;
options.homeostasisTau = 20;

numDendrites = options.numActive + options.numSilent;

% Set Up Correlation Situation
rHat = 20;
sigma = 0.5;
tau_c = 0.02/options.dt; % correlation tau

dendriteCorrPrm = [linspace(0.2,0,options.numActive),silentCorrelation*ones(1,options.numSilent)];
corrPrm = repmat(dendriteCorrPrm,options.numSynapses,1);
corrSigma = sqrt(sigma.^2 - corrPrm.^2);

keepTime = tic;
runsInEach = 1;
for runNum = 1:runsInEach
    % Setup new model
    iaf = buildIafCM(options);
    
    needInput = true;
    spikes = zeros(1,T);
    y = zeros(1,T);
    %vm = zeros(1,T);
    %econd = zeros(1,T);
    %icond = zeros(1,T);
    inputRates = zeros(iaf.numSynapses, numDendrites, T);
    
    % Downsample tracking 
    dsprm = 100;
    trackWeights = zeros(iaf.numSynapses, numDendrites, T/100);
    for t = 1:T
        % Print some updates to the screen
        if rem(t,T/100)==0
            %fprintf(repmat('\b',1,length(msg))); % -- can't do this on o2...
            msg = sprintf('Sample %d/%d seconds, %d Percent, EstTime: %.1f seconds ...\n',...
                t*options.dt,T*options.dt,100*t/T,toc(keepTime)*T/t*runsInEach/runNum);
            fprintf(msg);
        end

        if needInput
            interval = ceil(exprnd(tau_c))+1; % duration of interval for current rates
            y(t) = randn();
            x = corrSigma.*randn(iaf.numSynapses, iaf.numDendrites);
            inputRate = rHat * (1 + x + corrPrm*y(t));
            inputRate(inputRate<0)=0;
            trackInterval = interval - 1;
            if trackInterval>0
                needInput = false;
            end
        else
            y(t) = y(t-1);
            trackInterval = trackInterval - 1;
            if trackInterval==0
                needInput = true; 
            end
        end
        iaf = stepIafCM(iaf, inputRate);
        spikes(t) = iaf.spike;
        %vm(t) = iaf.vm;
        %econd(t) = sum(iaf.ampaConductance);
        %icond(t) = sum(iaf.gabaConductance);
        %inputRates(:,:,t) = inputRate;
        if rem(t,dsprm)==0
            trackWeights(:,:,round(t/dsprm)) = iaf.ampaWeights;
        end
    end
    %psth = sum(reshape(spikes,1000,T/1000),1);
    spkTimes = find(spikes);

    % Save data
    name = sprintf('iafCM_SD%d_SC%d_Run%d',sDepIdx,sCorIdx,(runIdx-1)*runsInEach+runNum);
    savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/songAbbott/cModel1',name);
    % savepath = fullfile('~/Documents/Research/o2/stdpModels/cf_withSilentData',name);
    %save(savepath,'iaf','spkTimes')
end

