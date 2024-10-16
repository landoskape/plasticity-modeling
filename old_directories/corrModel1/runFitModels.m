function [iaf,spkTimes,y,source,bPreSpikes,aPreSpikes] = runFitModels(iaf,T,verbose)
%function runo2_corrD(runIdx,apDepIdx,numStateIdx)
% run fit model without plasticity

rng('shuffle')

runsInEach = 1;
if ~exist('T','var'), T = 10*1000; end
if ~exist('verbose','var'), verbose = true; end


tauStim = round(0.01/iaf.dt); % time constant of stim (in samples)

for runNum = 1:runsInEach    
    needInput = true; % generate input signal
    y = zeros(iaf.numInputs,T);
    source = zeros(iaf.numSignals,T);
    spikes = zeros(1,T);
    bPreSpikes = zeros(iaf.numBasal,T);
    aPreSpikes = zeros(iaf.numApical,T);
    msg = ''; % Initialize msg
    keepTime = tic;
    for t = 1:T
        % Print some updates to the screen
        if verbose && rem(t,T/1000)==0
            timePer_t = toc(keepTime)/t;
            estTimeRemaining = (T-t)*timePer_t + T*timePer_t*(runsInEach-runNum);
            fprintf(repmat('\b',1,length(msg))); % -- can't do this on o2...
            msg = sprintf('Sample %.1f/%d seconds, %.1f Percent, EstTimeRemaining: %.1f seconds ...\n',...
                t*iaf.dt,T*iaf.dt,100*t/T,estTimeRemaining);
            fprintf(msg);
        end
        
        % Generate Input
        if needInput
            interval = ceil(exprnd(tauStim))+1; % duration of interval for current rates
            source(:,t) = randn(iaf.numSignals,1);
            input = (randn(1,iaf.numInputs) + source(:,t)'*iaf.sourceLoading)./iaf.varAdjustment;
            rate = iaf.rateStd * input + iaf.rateMean;
            rate(rate<0) = 0;
            y(:,t) = rate(:);
            trackInterval = interval - 1;
            if trackInterval>0
                needInput = false;
            end
        else
            y(:,t) = y(:,t-1); % ***** for saving input/updates *****
            source(:,t) = source(:,t-1); 
            trackInterval = trackInterval - 1;
            if trackInterval==0
                needInput = true;
            end
        end
        
        [iaf,bPreSpikes(:,t),aPreSpikes(:,t)] = stepFitModels(iaf,rate(:));
        spikes(t) = iaf.spike;
    end
    spkTimes = find(spikes);
end
