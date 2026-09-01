clear
close all
clc

% ============================================================
% GENERATE SYNTHETIC SUPERGRANULE-LIKE VELOCITY FIELD
%
% Creates
%
%   data/supergranule_synthetic.nc
%
% for use with
%
%   scripts/run_FTRCS_supergranule.jl
%
% The synthetic field is strongly compressible and consists of
% time-dependent Gaussian potential-flow cells with weaker
% rotational components.
%
% This script reconstructs a fixed realization from
%
%   matlab/supergranule_synthetic_parameters.mat
%
% so the generated field does not depend on MATLAB's random
% number generator.
%
% External NetCDF convention:
%
%   x,y       km
%   t         h
%   u,v       km/h
%
%   size(u) = size(v) = [Ny Nx Nt].
% ============================================================


% ============================================================
% REPOSITORY PATHS
% ============================================================

this_file = mfilename('fullpath');

if isempty(this_file)
    error('Run this script from a saved .m file.')
end

matlab_dir = fileparts(this_file);
repo_dir   = fileparts(matlab_dir);

parameter_file = fullfile( ...
    matlab_dir, ...
    'supergranule_synthetic_parameters.mat');

output_dir = fullfile( ...
    repo_dir, ...
    'data');

output_file = fullfile( ...
    output_dir, ...
    'supergranule_synthetic.nc');

if ~exist(output_dir,'dir')
    mkdir(output_dir)
end

if ~isfile(parameter_file)
    error( ...
        'Parameter file not found: %s', ...
        parameter_file)
end


% ============================================================
% LOAD FROZEN SYNTHETIC-FLOW PARAMETERS
% ============================================================

S = load(parameter_file);

Ncells = S.Ncells;

Nx = S.Nx;
Ny = S.Ny;

dx = S.dx;
dy = S.dy;
dt = S.dt;

t0 = S.t0;
t1 = S.t1;

xc = S.xc;
yc = S.yc;

sigma = S.sigma;

B0 = S.B0;
A0 = S.A0;

cx = S.cx;
cy = S.cy;

period = S.period;
phase  = S.phase;


% ============================================================
% GRID
% ============================================================

x = (0:Nx-1)' * dx;
y = (0:Ny-1)' * dy;
t = (t0:dt:t1)';

Nt = length(t);

[X,Y] = ndgrid(x,y);

Lx = x(end);
Ly = y(end);


% ============================================================
% SYNTHETIC VELOCITY MODEL
%
% Dominant potential component:
%
%       u_p = phi_x
%       v_p = phi_y
%
% Weaker rotational component:
%
%       u_r = -psi_y
%       v_r =  psi_x
%
% Total:
%
%       u = phi_x - psi_y
%       v = phi_y + psi_x
%
% For each cell,
%
%       phi = B_j(t) G_j
%       psi = A_j(t) G_j
%
% with
%
%       G_j =
%       exp(-[(x-x_j)^2+(y-y_j)^2]/(2 sigma_j^2)).
% ============================================================


% ============================================================
% STORAGE
%
% Internal construction:
%
%       [Nx Ny Nt] = [x y t]
% ============================================================

u = zeros(Nx,Ny,Nt,'single');
v = zeros(Nx,Ny,Nt,'single');


% ============================================================
% GENERATE VELOCITY FIELD
% ============================================================

fprintf('\nGenerating synthetic supergranule field\n');
fprintf('--------------------------------------\n');

fprintf('grid      = %d x %d x %d\n',Nx,Ny,Nt);
fprintf('dx,dy     = %.3f, %.3f km\n',dx,dy);
fprintf('dt        = %.6f h\n',dt);
fprintf('time      = %.2f -- %.2f h\n',t(1),t(end));
fprintf('cells     = %d\n',Ncells);

for k = 1:Nt

    tk = t(k);

    uk = zeros(Nx,Ny);
    vk = zeros(Nx,Ny);

    for j = 1:Ncells

        % ----------------------------------------------------
        % Slowly drifting center
        % ----------------------------------------------------

        xj = ...
            xc(j) + cx(j)*tk;

        yj = ...
            yc(j) + cy(j)*tk;

        % Keep centers inside the domain.

        xj = ...
            min(max(xj,0.05*Lx),0.95*Lx);

        yj = ...
            min(max(yj,0.05*Ly),0.95*Ly);


        % ----------------------------------------------------
        % Time-varying amplitude
        % ----------------------------------------------------

        modj = ...
            0.65 + ...
            0.35*cos( ...
                2*pi*tk/period(j) + ...
                phase(j));

        Bj = ...
            B0(j) * modj;

        Aj = ...
            A0(j) * modj;


        % ----------------------------------------------------
        % Gaussian cell
        % ----------------------------------------------------

        rx = ...
            X - xj;

        ry = ...
            Y - yj;

        s2 = ...
            sigma(j)^2;

        G = ...
            exp( ...
                -(rx.^2 + ry.^2) / ...
                (2*s2));


        % ----------------------------------------------------
        % Potential component
        %
        % phi = Bj G
        %
        % phi_x = -(Bj/sigma^2) rx G
        % phi_y = -(Bj/sigma^2) ry G
        % ----------------------------------------------------

        phi_x = ...
            -(Bj/s2) .* rx .* G;

        phi_y = ...
            -(Bj/s2) .* ry .* G;


        % ----------------------------------------------------
        % Rotational component
        %
        % psi = Aj G
        %
        % u_r = -psi_y
        % v_r =  psi_x
        % ----------------------------------------------------

        psi_x = ...
            -(Aj/s2) .* rx .* G;

        psi_y = ...
            -(Aj/s2) .* ry .* G;


        % ----------------------------------------------------
        % TOTAL VELOCITY
        % ----------------------------------------------------

        uk = ...
            uk + ...
            phi_x - psi_y;

        vk = ...
            vk + ...
            phi_y + psi_x;

    end

    u(:,:,k) = single(uk);
    v(:,:,k) = single(vk);

end


% ============================================================
% DIAGNOSTIC: VORTICITY + STREAMLINES
%
% Plot t = 0,2,4,6 h.
% ============================================================

tplot = [0 2 4 6];

figure( ...
    'Color','w', ...
    'Position',[100 100 1500 400]);

for n = 1:length(tplot)

    [~,k] = ...
        min(abs(t-tplot(n)));

    uk = ...
        double(u(:,:,k));

    vk = ...
        double(v(:,:,k));


    % --------------------------------------------------------
    % VORTICITY
    %
    %       omega = dv/dx - du/dy
    %
    % Arrays here are [Nx Ny], so
    %
    %       d/dx -> dimension 1
    %       d/dy -> dimension 2.
    % --------------------------------------------------------

    dudy = ...
        zeros(size(uk));

    dvdx = ...
        zeros(size(vk));


    % du/dy

    dudy(:,2:end-1) = ...
        ( ...
            uk(:,3:end) - ...
            uk(:,1:end-2) ...
        ) / (2*dy);

    dudy(:,1) = ...
        ( ...
            uk(:,2) - ...
            uk(:,1) ...
        ) / dy;

    dudy(:,end) = ...
        ( ...
            uk(:,end) - ...
            uk(:,end-1) ...
        ) / dy;


    % dv/dx

    dvdx(2:end-1,:) = ...
        ( ...
            vk(3:end,:) - ...
            vk(1:end-2,:) ...
        ) / (2*dx);

    dvdx(1,:) = ...
        ( ...
            vk(2,:) - ...
            vk(1,:) ...
        ) / dx;

    dvdx(end,:) = ...
        ( ...
            vk(end,:) - ...
            vk(end-1,:) ...
        ) / dx;


    omega = ...
        dvdx - dudy;


    % --------------------------------------------------------
    % PLOT
    % --------------------------------------------------------

    subplot(1,4,n)

    imagesc( ...
        x, ...
        y, ...
        omega');

    set(gca,'YDir','normal');

    axis equal
    axis tight

    hold on

    h = streamslice( ...
        X', ...
        Y', ...
        uk', ...
        vk', ...
        1.0);

    set( ...
        h, ...
        'Color','k', ...
        'LineWidth',0.5);

    colormap(redwhiteblue);

    title( ...
        sprintf('$t=%.0f$ h',t(k)), ...
        'Interpreter','latex');

    xlabel( ...
        '$x$ [km]', ...
        'Interpreter','latex');

    if n == 1

        ylabel( ...
            '$y$ [km]', ...
            'Interpreter','latex');

    end

    box on

end


% ============================================================
% EXTERNAL NETCDF CONVENTION
%
% Internal arrays:
%
%       [Nx Ny Nt] = [x y t]
%
% NetCDF arrays:
%
%       [Ny Nx Nt] = [y x time]
% ============================================================

u_nc = permute(u,[2 1 3]);
v_nc = permute(v,[2 1 3]);


% ============================================================
% SAVE NETCDF
% ============================================================

if exist(output_file,'file')
    delete(output_file)
end


% ------------------------------------------------------------
% COORDINATES
% ------------------------------------------------------------

nccreate( ...
    output_file, ...
    'x', ...
    'Dimensions',{'x',Nx}, ...
    'Datatype','double');

nccreate( ...
    output_file, ...
    'y', ...
    'Dimensions',{'y',Ny}, ...
    'Datatype','double');

nccreate( ...
    output_file, ...
    't', ...
    'Dimensions',{'time',Nt}, ...
    'Datatype','double');


% ------------------------------------------------------------
% VELOCITY
% ------------------------------------------------------------

nccreate( ...
    output_file, ...
    'u', ...
    'Dimensions', ...
    {'y',Ny, ...
     'x',Nx, ...
     'time',Nt}, ...
    'Datatype','single');

nccreate( ...
    output_file, ...
    'v', ...
    'Dimensions', ...
    {'y',Ny, ...
     'x',Nx, ...
     'time',Nt}, ...
    'Datatype','single');


% ------------------------------------------------------------
% WRITE DATA
% ------------------------------------------------------------

ncwrite( ...
    output_file, ...
    'x', ...
    x);

ncwrite( ...
    output_file, ...
    'y', ...
    y);

ncwrite( ...
    output_file, ...
    't', ...
    t);

ncwrite( ...
    output_file, ...
    'u', ...
    u_nc);

ncwrite( ...
    output_file, ...
    'v', ...
    v_nc);


% ============================================================
% NETCDF ATTRIBUTES
% ============================================================

ncwriteatt( ...
    output_file, ...
    'x', ...
    'units', ...
    'km');

ncwriteatt( ...
    output_file, ...
    'y', ...
    'units', ...
    'km');

ncwriteatt( ...
    output_file, ...
    't', ...
    'units', ...
    'h');

ncwriteatt( ...
    output_file, ...
    'u', ...
    'units', ...
    'km/h');

ncwriteatt( ...
    output_file, ...
    'v', ...
    'units', ...
    'km/h');

ncwriteatt( ...
    output_file, ...
    '/', ...
    'velocity_storage_convention', ...
    '[Ny Nx Nt] = [y x time]');

ncwriteatt( ...
    output_file, ...
    '/', ...
    'description', ...
    [ ...
        'Synthetic compressible supergranule-like velocity field: ' ...
        'dominant Gaussian potential cells plus weaker rotational ' ...
        'Gaussian components.' ...
    ]);


% ============================================================
% FINAL CHECK
% ============================================================

fprintf('\nSaved synthetic velocity field\n');
fprintf('------------------------------\n');
fprintf('%s\n',output_file);

info = ...
    ncinfo(output_file);

fprintf('\nNetCDF variables\n');
fprintf('----------------\n');

for k = 1:length(info.Variables)

    fprintf( ...
        '%-4s  size = %s\n', ...
        info.Variables(k).Name, ...
        mat2str(info.Variables(k).Size));

end