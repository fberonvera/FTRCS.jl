clear
close all
clc

% ============================================================
% OUTPUT
% ============================================================

output_dir = fullfile( ...
    getenv('HOME'), ...
    'Documents','Julia','ftrcs','data');

output_file = fullfile( ...
    output_dir, ...
    'hinode_synthetic.nc');

if ~exist(output_dir,'dir')
    mkdir(output_dir)
end


% ============================================================
% GRID
% ============================================================

Nx = 432;
Ny = 432;

dx = 116.0;               % km
dy = 116.0;               % km

dt = 90.0 / 3600.0;       % h

t0 = 0.0;
t1 = 12.0;

x = (0:Nx-1)' * dx;
y = (0:Ny-1)' * dy;
t = (t0:dt:t1)';

Nt = length(t);

[X,Y] = ndgrid(x,y);


% ============================================================
% SYNTHETIC SUPERGRANULE PARAMETERS
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
% ============================================================

rng(1)

Ncells = 14;

Lx = x(end);
Ly = y(end);

% Cell centers.

xc = ...
    0.10*Lx + 0.80*Lx*rand(Ncells,1);

yc = ...
    0.10*Ly + 0.80*Ly*rand(Ncells,1);

% Typical cell widths.

sigma = ...
    2500 + 2500*rand(Ncells,1);   % km

% Potential amplitudes.
%
% Units chosen so resulting velocity is km/h.

B0 = ...
    1500 + 2500*rand(Ncells,1);

B0 = ...
    B0 .* sign(randn(Ncells,1));

% Rotational amplitudes are deliberately weaker.

A0 = ...
    0.20 * B0;

% Slow center drift.

drift_speed = ...
    20 + 40*rand(Ncells,1);       % km/h

drift_angle = ...
    2*pi*rand(Ncells,1);

cx = ...
    drift_speed .* cos(drift_angle);

cy = ...
    drift_speed .* sin(drift_angle);

% Slow amplitude modulation.

period = ...
    4 + 8*rand(Ncells,1);         % h

phase = ...
    2*pi*rand(Ncells,1);


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
        % Gaussian
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
        % Potential part
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
        % Rotational part
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
% ============================================================

tplot = [0 2 4 6];

[X,Y] = ndgrid(x,y);

figure('Color','w');

for n = 1:length(tplot)

    [~,k] = min(abs(t-tplot(n)));

    uk = double(u(:,:,k));
    vk = double(v(:,:,k));

    % Vorticity: omega = dv/dx - du/dy

    dudy = zeros(size(uk));
    dvdx = zeros(size(vk));

    dudy(:,2:end-1) = ...
        (uk(:,3:end) - uk(:,1:end-2)) / (2*dy);

    dudy(:,1) = ...
        (uk(:,2) - uk(:,1)) / dy;

    dudy(:,end) = ...
        (uk(:,end) - uk(:,end-1)) / dy;

    dvdx(2:end-1,:) = ...
        (vk(3:end,:) - vk(1:end-2,:)) / (2*dx);

    dvdx(1,:) = ...
        (vk(2,:) - vk(1,:)) / dx;

    dvdx(end,:) = ...
        (vk(end,:) - vk(end-1,:)) / dx;

    omega = dvdx - dudy;

    subplot(1,4,n)

    imagesc(x,y,omega');

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

    set(h, ...
        'Color','k', ...
        'LineWidth',0.5);

    colormap(redwhiteblue);

    title( ...
        sprintf('$t=%.0f$ h',t(k)), ...
        'Interpreter','latex');

    xlabel('$x$ [km]', ...
        'Interpreter','latex');

    if n == 1
        ylabel('$y$ [km]', ...
            'Interpreter','latex');
    end

    box on

end


% ============================================================
% EXTERNAL NETCDF CONVENTION
%
% Store as
%
%       [Ny Nx Nt] = [y x time].
% ============================================================

u_nc = permute(u,[2 1 3]);
v_nc = permute(v,[2 1 3]);


% ============================================================
% SAVE NETCDF
% ============================================================

if exist(output_file,'file')
    delete(output_file)
end

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

nccreate( ...
    output_file, ...
    'u', ...
    'Dimensions', ...
    {'y',Ny,'x',Nx,'time',Nt}, ...
    'Datatype','single');

nccreate( ...
    output_file, ...
    'v', ...
    'Dimensions', ...
    {'y',Ny,'x',Nx,'time',Nt}, ...
    'Datatype','single');

ncwrite(output_file,'x',x);
ncwrite(output_file,'y',y);
ncwrite(output_file,'t',t);

ncwrite(output_file,'u',u_nc);
ncwrite(output_file,'v',v_nc);

ncwriteatt(output_file,'x','units','km');
ncwriteatt(output_file,'y','units','km');
ncwriteatt(output_file,'t','units','h');

ncwriteatt(output_file,'u','units','km/h');
ncwriteatt(output_file,'v','units','km/h');

ncwriteatt( ...
    output_file, ...
    '/', ...
    'velocity_storage_convention', ...
    '[Ny Nx Nt] = [y x time]');

ncwriteatt( ...
    output_file, ...
    '/', ...
    'description', ...
    ['Synthetic compressible supergranule-like velocity field: ' ...
     'dominant Gaussian potential cells plus weaker rotational ' ...
     'Gaussian components.']);

fprintf('\nSaved\n');
fprintf('-----\n');
fprintf('%s\n',output_file);

% ============================================================
% SAVE SYNTHETIC-FLOW PARAMETERS
%
% Freeze this particular realization so the distributed
% generator does not depend on MATLAB's random-number generator.
% ============================================================

parameter_file = fullfile( ...
   fileparts(mfilename('fullpath')), ...
   'supergranule_synthetic_parameters.mat');

save( ...
   parameter_file, ...
   'Ncells', ...
   'Nx','Ny', ...
   'dx','dy','dt', ...
   't0','t1', ...
   'xc','yc', ...
   'sigma', ...
   'B0','A0', ...
   'cx','cy', ...
   'period','phase');

fprintf('\nSaved synthetic-flow parameters\n');
fprintf('--------------------------------\n');
fprintf('%s\n',parameter_file);