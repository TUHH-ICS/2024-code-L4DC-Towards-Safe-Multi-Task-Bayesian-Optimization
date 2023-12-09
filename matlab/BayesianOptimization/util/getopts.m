%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Author(s): Jannis Lübsen
%--------------------------------------------------------------------------------------------
% Merges the user options with initial options.

function opts=getopts(opts,newOpts)
if isempty(newOpts), return; end
ch1=fieldnames(newOpts);
ch2=fieldnames(opts);
for i=1:length(ch1)
    for j=1:length(ch2)
        if strcmp(ch1{i},ch2{j})
            if isstruct(opts.(ch2{j})) && isstruct(newOpts.(ch1{i}))
                opts.(ch2{j})=getopts(opts.(ch2{j}),newOpts.(ch1{i}));
            else
                opts.(ch2{j})=newOpts.(ch1{i});
            end
            break;
        end
        if j == length(ch2)
            opts.(ch1{i})=newOpts.(ch1{i});
        end
    end
end
end

