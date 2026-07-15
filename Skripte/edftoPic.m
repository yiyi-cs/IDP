%% =========================================================================
  % SMOOTH-PURSUIT-KALIBRIERUNG: VISUALISIERUNG DES BLICKVERLAUFS
% =========================================================================
  %
% Beschreibung:
  %   Dieses Skript visualisiert die mittels EyeLink 1000 erfassten 
%   Blickverläufe während der Smooth-Pursuit-Kalibrierung bei variierender
%   Zielgeschwindigkeit. Der zeitliche Verlauf wird durch einen 
%   Blau-Rot-Farbgradienten kodiert.
%
% Voraussetzungen:
  %   - MATLAB R2025a oder neuer
%   - Edf2Mat Toolbox (https://github.com/uzh/edf-converter)
%   - Visual C++ Redistributable 2015 und 2008
%
% Eingabe:
  %   - EDF-Dateien im Ordner: [top_folder]/kly[1-8]/kly[1-8]*.edf
%
% Ausgabe:
  %   - 4 PNG-Dateien (DIN A4, 300 dpi): kly_seite[1-4].png
%
% Autor: [Name]
% Datum: Februar 2026
% Version: 1.0
%
% Zugehörige Arbeit:
  %   Pilotstudie zur CV-basierten Blickerfassung für die FVE-basierte 
%   Neglect-Diagnostik
%
% =========================================================================
  
  %% === PAKETE UND ABHÄNGIGKEITEN ===
    % Edf2Mat muss im MATLAB-Pfad sein:
    %   addpath('C:\...\edf-converter-master');
  %   savepath;
  
  % Prüfe ob Edf2Mat verfügbar ist
  if ~exist('Edf2Mat', 'class')
  error(['Edf2Mat Toolbox nicht gefunden. ', ...
         'Bitte installieren: https://github.com/uzh/edf-converter']);
  end
  
  %% === KONFIGURATION ===
    top_folder = 'C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse\kly';
  output_folder = top_folder;
  
  file_info = struct(...
                     'speed', {400, 300, 500, 200, 200, 500, 300, 400}, ...
                     'reverse', {false, false, false, false, true, true, true, true});
  
  %% === BILDSCHIRM-PARAMETER ===
    res_x = 1920;
  res_y = 1080;
  margin_x = res_x * 0.17;
  margin_y = res_y * 0.20;
  
  calc_pos = @(col, row) [margin_x + (res_x - 2*margin_x) * col, ...
                          margin_y + (res_y - 2*margin_y) * row];
  
  % Kalibrierungspunkte
  points = [
    res_x/2, res_y/2;
    calc_pos(0.0, 0.0);
    calc_pos(0.5, 0.0);
    calc_pos(1.0, 0.0);
    calc_pos(1.0, 0.5);
    calc_pos(0.5, 0.5);
    calc_pos(0.0, 0.5);
    calc_pos(0.0, 1.0);
    calc_pos(0.5, 1.0);
    calc_pos(1.0, 1.0)
  ];
  
  center_x = res_x / 2;
  center_y = res_y / 2;
  
  % Achsengrenzen definieren
  x_lim = [-50 2000];
  y_lim = [-50 1150];
  
  %% === HAUPTSCHLEIFE: 4 SEITEN ERSTELLEN ===
    for page = 1:4
  % DIN A4 Hochformat
  fig = figure('Units', 'centimeters', 'Position', [1 1 21 29.7], ...
               'Color', 'w', 'PaperUnits', 'centimeters', ...
               'PaperSize', [21 29.7], 'PaperPosition', [0 0 21 29.7]);
  
  file_indices = [(page-1)*2 + 1, (page-1)*2 + 2];
  
  for subplot_idx = 1:2
  file_number = file_indices(subplot_idx);
  
  %% --- EDF-Datei finden und laden ---
    subfolder = fullfile(top_folder, sprintf('kly%d', file_number));
  edf_files = dir(fullfile(subfolder, sprintf('kly%d*.edf', file_number)));
  
  if isempty(edf_files)
  warning('Keine EDF-Datei gefunden für kly%d', file_number);
  continue;
  end
  
  edf_path = fullfile(subfolder, edf_files(1).name);
  fprintf('Lade: %s\n', edf_path);
  
  edf_data = Edf2Mat(edf_path);
  
  %% --- Sequenz bestimmen ---
    current_reverse = file_info(file_number).reverse;
  current_speed = file_info(file_number).speed;
  
  if current_reverse
  seq = [1, 10, 9, 8, 7, 6, 5, 4, 3, 2];
  direction_str = 'Start: rechts unten';
  else
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  direction_str = 'Start: links oben';
  end
  
  target_x = points(seq, 1);
  target_y = points(seq, 2);
  
  %% --- Blickdaten extrahieren ---
    x = edf_data.Samples.posX;
  y = edf_data.Samples.posY;
  t = edf_data.Samples.time;
  t_norm = (t - min(t)) / (max(t) - min(t));
  
  % Nur Punkte im sichtbaren Bereich
  valid = x >= x_lim(1) & x <= x_lim(2) & y >= y_lim(1) & y <= y_lim(2);
  x = x(valid);
  y = y(valid);
  t_norm = t_norm(valid);
  
  %% --- Subplot erstellen ---
    if subplot_idx == 1
  ax_pos = [0.08 0.51 0.82 0.44];
  else
    ax_pos = [0.08 0.05 0.82 0.44];
  end
  axes('Position', ax_pos);
  
  % Bildschirmrahmen
  rectangle('Position', [0, 0, res_x, res_y], ...
            'EdgeColor', [0.6 0.6 0.6], 'LineWidth', 1);
  hold on;
  
  % Blickverlauf
  scatter(x, y, 4, t_norm, 'filled', 'MarkerFaceAlpha', 0.7);
  
  % Kalibrierungspunkte (70% transparent, schwarze Zahlen)
center_already_drawn = false;
for i = 1:length(target_x)
is_center = (abs(target_x(i) - center_x) < 1) && (abs(target_y(i) - center_y) < 1);

if is_center && ~center_already_drawn
scatter(target_x(i), target_y(i), 80, 'o', ...
        'MarkerEdgeColor', [0.7 0.7 0.7], ...
        'MarkerFaceColor', 'w', ...
        'MarkerFaceAlpha', 0.3, ...
        'LineWidth', 0.5);
text(target_x(i), target_y(i), '0/5', ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
     'FontSize', 6, 'Color', 'k');
center_already_drawn = true;
elseif ~is_center
scatter(target_x(i), target_y(i), 80, 'o', ...
        'MarkerEdgeColor', [0.7 0.7 0.7], ...
        'MarkerFaceColor', 'w', ...
        'MarkerFaceAlpha', 0.3, ...
        'LineWidth', 0.5);
text(target_x(i), target_y(i), sprintf('%d', i-1), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
     'FontSize', 6, 'Color', 'k');
end
end

% Farbskala
colormap(turbo);
cb = colorbar;
cb.Label.String = 'Zeit (normalisiert)';
cb.Label.FontSize = 8;
cb.Ticks = [0 0.5 1];
caxis([0 1]);

% Achsen
set(gca, 'YDir', 'reverse');
axis equal;
xlim(x_lim);
ylim(y_lim);
xlabel('X (px)', 'FontSize', 9);
ylabel('Y (px)', 'FontSize', 9);
set(gca, 'FontSize', 8, 'Box', 'off');

% Titel
title(sprintf('%d px/s | %s', current_speed, direction_str), ...
      'FontSize', 10, 'FontWeight', 'normal');
end

%% --- Seite speichern ---
  output_file = fullfile(output_folder, sprintf('kly_seite%d.png', page));
exportgraphics(fig, output_file, 'Resolution', 300);
fprintf('Gespeichert: %s\n', output_file);

close(fig);
end

fprintf('\n=== FERTIG! 4 Seiten exportiert ===\n');
