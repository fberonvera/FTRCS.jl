clear
close all
clc

% ============================================================
% INPUT
% ============================================================

ncfile = fullfile( ...
    getenv('HOME'), ...
    'Documents','Julia','ftrcs','runs', ...
    'FTRCS_supergranule_synthetic_output.nc');


% ============================================================
% READ JULIA OUTPUT
% ============================================================

x = ncread(ncfile,'x');
y = ncread(ncfile,'y');
t = ncread(ncfile,'time');

A = logical( ...
    ncread(ncfile,'candidate_mask') ...
    );

rho = ncread( ...
    ncfile,'candidate_rho' ...
    );

object = ncread( ...
    ncfile,'candidate_seba_index' ...
    );

E = ncread( ...
    ncfile,'E_LAVD' ...
    );

Nc = size(A,4);

isFTRCS = E > 1.0;


% ============================================================
% REPORT
% ============================================================

fprintf('\nIDL-SEBA candidates\n');
fprintf('--------------------------------------------------\n');
fprintf('rank   SEBA object      rho        E_LAVD   FTRCS\n');

for ic = 1:Nc

    fprintf( ...
        '%4d   %8d   %.6e    %6.3f     %s\n', ...
        ic, ...
        object(ic), ...
        rho(ic), ...
        E(ic), ...
        yesno(isFTRCS(ic)) ...
        );

end

fprintf('\nFTRCS selected = %d / %d\n', ...
    nnz(isFTRCS),Nc);


% ============================================================
% TIME TO DISPLAY
% ============================================================

tplot = 0;

[~,kt] = min(abs(t-tplot));

% ============================================================
% DISTINCT CANDIDATE COLORS
% ============================================================

h = mod((0:Nc-1)' * 0.61803398875,1);

cmap = hsv2rgb( ...
    [h, ...
     0.75*ones(Nc,1), ...
     0.90*ones(Nc,1)] ...
    );


% ============================================================
% FIGURE
% ============================================================

figure( ...
    'Color','w' ...
    );

ax = axes;
hold(ax,'on');


% ============================================================
% FILLED FTRCS
%
% Fill only candidates satisfying E_LAVD > 1.
% ============================================================

for ic = 1:Nc

    if ~isFTRCS(ic)
        continue
    end

    mask = A(:,:,kt,ic);

    if ~any(mask(:))
        continue
    end

    rgb = zeros(length(y),length(x),3);

    for q = 1:3
        rgb(:,:,q) = cmap(ic,q);
    end

    him = image( ...
        ax, ...
        x, ...
        y, ...
        rgb ...
        );

    set( ...
        him, ...
        'AlphaData',0.40*double(mask'), ...
        'AlphaDataMapping','none' ...
        );

end


% ============================================================
% CONTOURS OF ALL IDL-SEBA CANDIDATES
%
% All candidates are outlined, irrespective of LAVD.
% ============================================================

for ic = 1:Nc

    mask = A(:,:,kt,ic);

    if ~any(mask(:))
        continue
    end

    contour( ...
        ax, ...
        x, ...
        y, ...
        double(mask)', ...
        [0.5 0.5], ...
        'Color',cmap(ic,:), ...
        'LineWidth',1.4 ...
        );

end


% ============================================================
% AXES
% ============================================================

set(ax,'YDir','normal');

axis(ax,'equal');
axis(ax,'tight');

xlim( ...
    ax, ...
    [min(x) max(x)] ...
    );

ylim( ...
    ax, ...
    [min(y) max(y)] ...
    );

box(ax,'on');

xlabel( ...
    ax, ...
    '$x$ (km)', ...
    'Interpreter','latex' ...
    );

ylabel( ...
    ax, ...
    '$y$ (km)', ...
    'Interpreter','latex' ...
    );

title( ...
    ax, ...
    sprintf('$t = %.1f$ h',t(kt)), ...
    'Interpreter','latex' ...
    );


% ============================================================
% LOCAL FUNCTION
% ============================================================

function s = yesno(q)

if q
    s = 'yes';
else
    s = 'no';
end

end