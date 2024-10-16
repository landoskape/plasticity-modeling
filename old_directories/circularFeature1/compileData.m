

% script to analyze results from a run where we varied tuning width
lmPath = '/Users/landauland/Documents/Research/SabatiniLab/presentations/LabMeeting/200417';
hpath = '/Users/landauland/Documents/Research/o2/stdpModels/circularFeature1';


% Simulation Parameters
NS = 200;
numRuns = 15;
NExc = 1200;
dt = 0.0001;
simTime = 10000;
nsamples = simTime/dt;

tuningSharpness=[0.1,1,2,3,5,10,0.3,0.5];
NT = length(tuningSharpness);
prmNames = 'tuningSharpness';
prmValues = tuningSharpness; % to keep in compiled .mat file

% Preprocessing Parameters
cnvTime = 10; 
cnvResolution = 0.01;
cnvSamples = cnvTime/cnvResolution;
cnvSampleOffset = (simTime-cnvTime)/dt;
cnvFullSamples = cnvTime/dt;
stTime = 0.05;
stSamples = stTime/dt;
spkRateSamples = nsamples*dt;

% Preallocation
prmLabel = cell(NT,1);
prmIndex = zeros(NT,1);
allWeights = zeros(NExc,NS,numRuns,NT);
spikeRate = zeros(spkRateSamples,numRuns,NT); % psth
cnvRate = zeros(cnvSamples,numRuns,NT); % convergent rate / stim
cnvStim = zeros(cnvFullSamples,numRuns,NT);
stAll = cell(numRuns,NT); % spike-triggered stimulus average
stAvg = zeros(stSamples,numRuns,NT);
stStd = zeros(stSamples,numRuns,NT);
for tidx = 1:NT
    prmLabel{tidx} = sprintf('TuningSharpness%d',tidx);
    prmIndex(tidx)=tidx;
    for ridx = 1:numRuns
        fprintf('Tuning %d/%d, Run %d/%d, ...\n',tidx,NT,ridx,numRuns);
        name = sprintf('OneFeatureModel_1_Tuning%d_Run%d.mat',tidx,ridx);
        try
            cdata = load(fullfile(hpath, 'data',name),'gA','spkTimes','stimValue');
            allWeights(:,:,ridx,tidx) = cdata.gA;
            cspk=false(nsamples,1);
            cspk(cdata.spkTimes)=true;
            cpsth = sum(reshape(cspk,nsamples/spkRateSamples,spkRateSamples),1)*nsamples/spkRateSamples*dt;
            spikeRate(:,ridx,tidx) = cpsth;
            % Measure convergent spikes in high res time window
            cnvSpkTimes = cdata.spkTimes(cdata.spkTimes>cnvSampleOffset);
            cnvSpk = false(cnvFullSamples,1);
            cnvSpk(cnvSpkTimes-cnvSampleOffset)=true;
            cnvpsth = sum(reshape(cnvSpk,cnvResolution/dt,cnvFullSamples/(cnvResolution/dt)),1)/cnvResolution;
            cnvRate(:,ridx,tidx)=cnvpsth;
            cnvStim(:,ridx,tidx)=cdata.stimValue(end-cnvFullSamples+1:end);
            % Get spk triggered stimulus average
            NCS = length(cnvSpkTimes);
            cstavg = zeros(stSamples,NCS);
            for ncs = 1:NCS
                cCnvSpk = cnvSpkTimes(ncs);
                cstavg(:,ncs)=cdata.stimValue(cCnvSpk-stSamples+1:cCnvSpk);
            end
            stAll{ridx,tidx}=cstavg;
            stAvg(:,ridx,tidx)=circ_mean(cstavg,[],2);
            stStd(:,ridx,tidx)=circ_std(cstavg,[],[],2);
        catch
            disp([name, ' failed...']);
        end
    end
end

compiledResults.prmNames=prmNames;
compiledResults.prmValues=prmValues;
compiledResults.prmLabel=prmLabel;
compiledResults.prmIndex=prmIndex;
compiledResults.allWeights=allWeights;
compiledResults.spikeRate=spikeRate;
compiledResults.cnvRate=cnvRate;
compiledResults.cnvStim=cnvStim;
compiledResults.stAll=stAll;
compiledResults.stAvg=stAvg;
compiledResults.stStd=stStd;

% save(fullfile(hpath,'circularFeature1_compiledResults.mat'),'compiledResults');















