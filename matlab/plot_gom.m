clear
close all
clc

% ============================================================
% INPUT
% ============================================================

ncfile = fullfile( ...
    getenv('HOME'), ...
    'Documents','Julia','ftrcs','runs', ...
    'FTRCS_gom_output.nc');

vel_file = fullfile( ...
    getenv('HOME'), ...
    'Documents','Julia','ftrcs','data', ...
    'aviso_20130518_20140421.nc');


% ============================================================
% ANALYSIS SPACE/TIME WINDOW
%
% Must match run_FTRCS_gom.jl exactly.
% ============================================================

lon_bounds = [-98.0 -80.0];
lat_bounds = [ 18.0  31.0];

analysis_t0 = datetime(2013,5,18);
analysis_days = 120;

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
% t = 0
% ============================================================

[~,kt] = min(abs(t));


% ============================================================
% READ AVISO VELOCITY
% ============================================================

lon_all = double(ncread(vel_file,'lon'));
lat_all = double(ncread(vel_file,'lat'));

time = double(ncread(vel_file,'time'));

time_datetime = ...
    datetime(1970,1,1) + days(time);

analysis_t1 = ...
    analysis_t0 + days(analysis_days);

ix = find( ...
    lon_all >= lon_bounds(1) & ...
    lon_all <= lon_bounds(2));

iy = find( ...
    lat_all >= lat_bounds(1) & ...
    lat_all <= lat_bounds(2));

it = find( ...
    time_datetime >= analysis_t0 & ...
    time_datetime <= analysis_t1);

lon = lon_all(ix);
lat = lat_all(iy);

u = double(ncread( ...
    vel_file, ...
    'u', ...
    [ix(1) iy(1) it(1)], ...
    [numel(ix) numel(iy) numel(it)]));

v = double(ncread( ...
    vel_file, ...
    'v', ...
    [ix(1) iy(1) it(1)], ...
    [numel(ix) numel(iy) numel(it)]));


% ============================================================
% LAND MASK
%
% Match run_FTRCS_gom.jl: retain the full rectangular domain
% and set missing AVISO velocities over land to zero.
% ============================================================

u(~isfinite(u)) = 0;
v(~isfinite(v)) = 0;


% ============================================================
% FIRST VELOCITY SLICE
% ============================================================

u0 = squeeze(u(:,:,1));
v0 = squeeze(v(:,:,1));


% ============================================================
% PROJECT VELOCITY GRID TO LOCAL CARTESIAN COORDINATES
%
% Same projection as run_FTRCS_gom.jl.
% ============================================================

Re = 6371.0;       % km

lat0 = mean(lat);

xv = ...
    Re*cosd(lat0)*deg2rad(lon-lon(1));

yv = ...
    Re*deg2rad(lat-lat(1));


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

figure('Color','w');

ax = axes;
hold(ax,'on');


% ============================================================
% VELOCITY QUIVER
% ============================================================

[XV,YV] = ndgrid(xv,yv);

stride = 3;

quiver( ...
    ax, ...
    XV(1:stride:end,1:stride:end), ...
    YV(1:stride:end,1:stride:end), ...
    u0(1:stride:end,1:stride:end), ...
    v0(1:stride:end,1:stride:end), ...
    1.5, ...
    'Color',[0.25 0.25 0.25], ...
    'LineWidth',0.6);


% ============================================================
% FILLED FTRCS
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
% LABEL IDL-SEBA OBJECTS
% ============================================================

for ic = 1:Nc

    mask = A(:,:,kt,ic);

    if ~any(mask(:))
        continue
    end

    [ii,jj] = find(mask);

    xc = mean(x(ii));
    yc = mean(y(jj));

    text( ...
        ax, ...
        xc, ...
        yc, ...
        sprintf('%d',object(ic)), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontWeight','bold', ...
        'FontSize',13, ...
        'Color','k', ...
        'BackgroundColor','w', ...
        'Margin',1 ...
        );

end


% ============================================================
% AXES
% ============================================================

set(ax,'YDir','normal');

axis(ax,'equal');
axis(ax,'tight');

xlim(ax,[min(x) max(x)]);
ylim(ax,[min(y) max(y)]);

box(ax,'on');

xlabel('$x$ (km)', ...
    'Interpreter','latex');

ylabel('$y$ (km)', ...
    'Interpreter','latex');

title('$t = 0$', ...
    'Interpreter','latex');