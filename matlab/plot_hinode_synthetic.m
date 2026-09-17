% plot_hinode.m
%
% Hinode finite-lifetime FTRCS from subpartition-of-unity SEBA supports: 16-panel summary figure.
%
% One 4 x 4 figure:
%   16 equally spaced times spanning the loaded FTRCS interval.
%   Instantaneous velocity vectors are light gray.
%   Rotational finite-lifetime coherent structures (FTRCS), classified from PU-defined FTCS episodes, are
%   shown only while their retained episode is alive.
%
% Run from the matlab/ directory.

clear
clc
close all

%% ------------------------------------------------------------
% Files
% -------------------------------------------------------------

hinode_file = '../data/hinode_synthetic.nc';

ftrcs_file = ...
   '../runs/FTRCS_hinode_synthetic_00h_12h_output.nc';

%% ------------------------------------------------------------
% Plot controls
% -------------------------------------------------------------

% Panel times are set automatically after reading the FTRCS
% time coordinate:
%
%     panel_times = linspace(t0,tfinal,16)
%
% Hence a 0--6 h run displays 16 equally spaced panels over
% the complete 0--6 h interval.

quiver_stride = 12;
quiver_scale = 1.5;
quiver_linewidth = 0.55;

ftrcs_alpha = 0.25;
ftrcs_linewidth = 1.25;

% Plot only rotational episodes.
plot_nonrotational_ftcs = false;

%% ------------------------------------------------------------
% Hinode coordinates and time
%
% hinode.nc:
%   x : [432]
%   y : [432]
%   t : [877]
%   u : [y x time]
%   v : [y x time]
% -------------------------------------------------------------

xh = double(ncread(hinode_file,'x'));
yh = double(ncread(hinode_file,'y'));
th = double(ncread(hinode_file,'t'));

xh = xh(:);
yh = yh(:);
th = th(:);

% Convert horizontal coordinates to Mm if stored in km.
if max(abs(xh)) > 1000
   xh_plot = xh/1000;
   yh_plot = yh/1000;
else
   xh_plot = xh;
   yh_plot = yh;
end

%% ------------------------------------------------------------
% FTRCS grid and masks
%
% Step-4 output:
%   candidate_mask : [x y time candidate]
%   time           : [49]
% -------------------------------------------------------------

xf = double(ncread(ftrcs_file,'x'));
yf = double(ncread(ftrcs_file,'y'));
tf = double(ncread(ftrcs_file,'time'));

xf = xf(:);
yf = yf(:);
tf = tf(:);

% ------------------------------------------------------------
% Panel layout
% -------------------------------------------------------------
nrows = 5;
ncols = 5;
npanels = nrows*ncols;
panel_times = ...
   linspace(tf(1),tf(end),npanels);


if max(abs(xf)) > 1000
   xf_plot = xf/1000;
   yf_plot = yf/1000;
else
   xf_plot = xf;
   yf_plot = yf;
end

candidate_mask = logical( ...
   ncread(ftrcs_file,'candidate_mask'));

candidate_seba_index = double( ...
   ncread(ftrcs_file,'candidate_seba_index'));

%% ------------------------------------------------------------
% Episode-level Step-4 classification
% -------------------------------------------------------------

ep_object = double( ...
   ncread(ftrcs_file,'episode_seba_index'));

ep_number = double( ...
   ncread(ftrcs_file,'episode_number'));

ep_birth_idx = double( ...
   ncread(ftrcs_file,'episode_birth_index'));

ep_death_idx = double( ...
   ncread(ftrcs_file,'episode_death_index'));

ep_birth = double( ...
   ncread(ftrcs_file,'episode_birth_time'));

ep_death = double( ...
   ncread(ftrcs_file,'episode_death_time'));

ep_slices = double( ...
   ncread(ftrcs_file,'episode_active_slices'));

ep_area_cells = double( ...
   ncread(ftrcs_file,'episode_birth_area_cells'));

ep_E = double( ...
   ncread(ftrcs_file,'episode_E_LAVD'));

ep_FTRCS = logical( ...
   ncread(ftrcs_file,'episode_FTRCS'));

%% ------------------------------------------------------------
% Physical birth areas
% -------------------------------------------------------------

dx_f_Mm = mean(diff(xf_plot));
dy_f_Mm = mean(diff(yf_plot));

cell_area_Mm2 = dx_f_Mm*dy_f_Mm;

ep_area_Mm2 = ...
   ep_area_cells*cell_area_Mm2;

%% ------------------------------------------------------------
% Map SEBA object number -> candidate-mask index
%
% IMPORTANT:
% The NetCDF candidate dimension is stored in Rayleigh-rank order,
% not original SEBA-object order.
%
% candidate_seba_index(ic) gives the actual SEBA object number
% associated with candidate_mask(:,:,:,ic).
%
% Never interpret candidate slot ic itself as the SEBA object number.
% -------------------------------------------------------------

Nc = numel(candidate_seba_index);

object_to_candidate = containers.Map( ...
   'KeyType','double', ...
   'ValueType','double');

for ic = 1:Nc
   object_to_candidate(candidate_seba_index(ic)) = ic;
end

% Every episode object must map to exactly one saved candidate.
if numel(unique(candidate_seba_index)) ~= Nc
   error('candidate_seba_index contains duplicate SEBA-object numbers.')
end

if any(~ismember(ep_object,candidate_seba_index))
   error(['At least one episode_seba_index has no matching ' ...
          'candidate_seba_index entry.'])
end

%% ------------------------------------------------------------
% Candidate-slot / SEBA-object consistency diagnostic
% -------------------------------------------------------------

fprintf('\n')
fprintf('NetCDF candidate mapping\n')
fprintf('------------------------\n')
fprintf('candidate slot   SEBA object\n')

for ic = 1:Nc
   fprintf('%8d        %8d\n',ic,candidate_seba_index(ic))
end

%% ------------------------------------------------------------
% Console summary: rotational episodes
% -------------------------------------------------------------

fprintf('\n')
fprintf('Hinode finite-lifetime FTRCS\n')
fprintf('--------------------------------------------------------------------------\n')
fprintf(['object  episode   birth[h]   death[h]   lifetime[h]   ' ...
         'E_LAVD   area_birth[Mm^2]\n'])

ii = find(ep_FTRCS);

for q = 1:numel(ii)

   ie = ii(q);

   fprintf( ...
      '%6d  %7d   %8.2f   %8.2f      %8.2f    %6.3f       %8.3f\n', ...
      ep_object(ie), ...
      ep_number(ie), ...
      ep_birth(ie), ...
      ep_death(ie), ...
      ep_death(ie)-ep_birth(ie), ...
      ep_E(ie), ...
      ep_area_Mm2(ie))

end

fprintf('\nFTRCS selected = %d / %d retained FTCS episodes\n', ...
   nnz(ep_FTRCS),numel(ep_FTRCS))

%% ------------------------------------------------------------
% Color assignment
%
% Persistent color for each rotational EPISODE, not merely for
% each SEBA object. Thus object 9 episode 1 and object 9 episode 2
% can be distinguished if they appear in the same diagnostic.
% -------------------------------------------------------------

Nrot = nnz(ep_FTRCS);

if Nrot > 0
   cmap = lines(Nrot);
else
   cmap = zeros(0,3);
end

episode_color = nan(numel(ep_FTRCS),1);

episode_color(ii) = 1:Nrot;

%% ------------------------------------------------------------
% Figure
% -------------------------------------------------------------

figure(1)
clf

set(gcf, ...
   'Color','w', ...
   'Units','normalized', ...
   'Position',[0.03 0.04 0.94 0.88])

tl = tiledlayout(nrows,ncols, ...
   'TileSpacing','compact', ...
   'Padding','compact');

fprintf('\nFTRCS population at displayed panels\n')
fprintf('---------------------------------\n')
fprintf(' time[h]   number alive   episodes\n')

for ip = 1:numel(panel_times)

   tp = panel_times(ip);

   ax = nexttile(tl,ip);

   hold(ax,'on')
   box(ax,'on')

   %% ---------------------------------------------------------
   % Instantaneous Hinode velocity nearest requested hour
   % ----------------------------------------------------------

   [~,kh] = min(abs(th-tp));

   % NetCDF variable dimensions are [y x time].
   % Read one instantaneous [Ny Nx] field only.
   uh = double( ...
      ncread( ...
         hinode_file, ...
         'u', ...
         [1 1 kh], ...
         [numel(yh) numel(xh) 1]));

   vh = double( ...
      ncread( ...
         hinode_file, ...
         'v', ...
         [1 1 kh], ...
         [numel(yh) numel(xh) 1]));

   uh = squeeze(uh);
   vh = squeeze(vh);

   % Quiver background.  Subsample the 432 x 432 Hinode
   % velocity grid to keep the arrows readable in a 4 x 4
   % panel layout.
   iq = 1:quiver_stride:numel(xh_plot);
   jq = 1:quiver_stride:numel(yh_plot);

   quiver( ...
      ax, ...
      xh_plot(iq), ...
      yh_plot(jq), ...
      uh(jq,iq), ...
      vh(jq,iq), ...
      quiver_scale, ...
      'Color',[0.65 0.65 0.65], ...
      'LineWidth',quiver_linewidth)

   %% ---------------------------------------------------------
   % Nearest IDL slice
   % ----------------------------------------------------------

   [~,kf] = min(abs(tf-tp));

   % An episode is shown only if it is rotational and the
   % displayed IDL time lies inside its retained lifespan.
   alive = ...
      ep_FTRCS & ...
      (tf(kf) >= ep_birth-1e-10) & ...
      (tf(kf) <= ep_death+1e-10);

   if plot_nonrotational_ftcs
      alive = ...
         (tf(kf) >= ep_birth-1e-10) & ...
         (tf(kf) <= ep_death+1e-10);
   end

   ia = find(alive);

   labels = strings(numel(ia),1);

   for q = 1:numel(ia)

      ie = ia(q);

      obj = ep_object(ie);

      if ~isKey(object_to_candidate,obj)
         warning( ...
            'SEBA object %d has no candidate-mask entry.', ...
            obj)
         continue
      end

      ic = object_to_candidate(obj);

      mask = candidate_mask(:,:,kf,ic);

      % The episode lifespan and the actual PU support must agree.
      % If an episode is declared alive but its saved support is
      % empty at this slice, flag the inconsistency explicitly.
      if ~any(mask(:))
         warning( ...
            ['Episode %d.%d is alive at t = %.3f h, but its ' ...
             'candidate_mask is empty at the nearest IDL slice.'], ...
            ep_object(ie),ep_number(ie),tf(kf))
         continue
      end

      % candidate_mask is stored [x y ...], whereas MATLAB
      % plotting with x and y vectors expects rows=y, columns=x.
      mask_plot = double(mask.');

      if ep_FTRCS(ie)

         c = cmap(episode_color(ie),:);

         % Filled translucent region.
         hp = imagesc( ...
            ax, ...
            xf_plot, ...
            yf_plot, ...
            mask_plot);

         set(hp, ...
            'AlphaData', ...
            ftrcs_alpha*mask_plot)

         % Give the binary overlay a constant episode color.
         % True pixels map to the chosen RGB value by using an
         % RGB image rather than the axes colormap.
         rgb = zeros( ...
            size(mask_plot,1), ...
            size(mask_plot,2), ...
            3);

         for jc = 1:3
            rgb(:,:,jc) = c(jc);
         end

         set(hp,'CData',rgb)

         % Boundary.
         contour( ...
            ax, ...
            xf_plot, ...
            yf_plot, ...
            mask_plot, ...
            [0.5 0.5], ...
            'Color',c, ...
            'LineWidth',ftrcs_linewidth)

      else

         contour( ...
            ax, ...
            xf_plot, ...
            yf_plot, ...
            mask_plot, ...
            [0.5 0.5], ...
            'Color',[0.35 0.35 0.35], ...
            'LineWidth',0.8)

      end

      labels(q) = ...
         sprintf('%d.%d', ...
            ep_object(ie), ...
            ep_number(ie));

   end

   labels = labels(labels ~= "");

   fprintf('%7.1f       %3d        ',tp,numel(labels))

   if isempty(labels)
      fprintf('--\n')
   else
      fprintf('%s\n',strjoin(cellstr(labels),', '))
   end

   %% ---------------------------------------------------------
   % Axes
   % ----------------------------------------------------------

   %axis(ax,'equal')

   xlim(ax,[min(xh_plot) max(xh_plot)])
   ylim(ax,[min(yh_plot) max(yh_plot)])

   title(ax, ...
      sprintf('$t=%g$ h',tp), ...
      'Interpreter','latex', ...
      'FontWeight','normal')

   set(ax, ...
      'FontSize',10, ...
      'Layer','top', ...
      'TickDir','out')

   % Keep the compact 4x4 panel arrangement readable.
   if ip <= 9
      set(ax,'XTickLabel',[])
   else
      xlabel(ax,'$x$ [Mm]', ...
         'Interpreter','latex')
   end

   col = mod(ip-1,4)+1;

   if col ~= 1
      set(ax,'YTickLabel',[])
   else
      ylabel(ax,'$y$ [Mm]', ...
         'Interpreter','latex')
   end

end

