%% elliptical grid score
% EG 24
function out = compute_elliptical_grid_score(rate_map,r_threshold_central,r_threshold_outer)
out.ell_grid_score = nan;
out.eccentricity = nan;
out.ellipticity = nan;

out.scale_matrix = nan(3,3);
out.Rm = nan;
out.rm = nan;
out.r = nan;

out.rotation_offset = nan;

out.scaled_xcorr = nan(size(spatial_autocorr(rate_map)));
out.rotated_xcorr = nan(size(spatial_autocorr(rate_map)));

out.central_peak = nan;
out.central_peak_radius = nan;
out.original_outer_peaks_pos = nan;
out.original_outer_peaks_dist = nan;
out.rotated_peaks_pos = nan;

out.failed_elliptical_test = 0;

spat_xcorr = rate_map; % WR: autocorrelograms are directly input to the function
% figure; imagesc(spat_xcorr); colorbar;

clean_xcorr = spat_xcorr;           % threshold and clean autocorrelogram
clean_xcorr(isnan(clean_xcorr) | isinf(clean_xcorr)) = 0;

col_len = size(clean_xcorr, 2);     % get size for rows / columns
row_len = size(clean_xcorr, 1);
mid_point = [col_len/2, row_len/2];

%% 1. find central and outer peaks
% find and take out central peak
clean_xcorr(clean_xcorr < r_threshold_central) = 0;

% find central peak 
xcorr_out_central = regionprops(logical(clean_xcorr),'Centroid', 'EquivDiameter', 'Circularity', 'Orientation'); 
central_peak = cell2mat({xcorr_out_central.Centroid}.'); 

if ~isempty(central_peak)
    % find central peak
    eu_dist = sqrt(sum((central_peak-mid_point).^2,2));         % compute euclidean distance of each peak from the centre 
    [~,mindx] = nanmin(eu_dist);  
    central_peak = xcorr_out_central(mindx).Centroid;           % closest is central peak
    central_peak_radius = xcorr_out_central(mindx).EquivDiameter/2;
else
    central_peak_radius = 1; 
    central_peak = [10, 10];
end

% save variables
out.central_peak = central_peak;
out.central_peak_radius = central_peak_radius;

% before finding outer ring, take out central peak
dist_mat = zeros(size(clean_xcorr)); 
dist_mat(ceil(central_peak(2)),ceil(central_peak(1))) = 1;     
dist_mat = bwdist(dist_mat); 
outer_xcorr = spat_xcorr;
outer_xcorr(dist_mat<ceil(central_peak_radius)+1) = 0; 

% keyboard
% find outer peaks
outer_xcorr(outer_xcorr < r_threshold_outer) = 0;
outer_xcorr(isnan(outer_xcorr) | isinf(outer_xcorr)) = 0;
xcorr_out = regionprops(logical(outer_xcorr),'Centroid', 'EquivDiameter', 'Circularity', 'Orientation'); 
peak_coords = cell2mat({xcorr_out.Centroid}.'); 

% if ~isempty(peak_coords)
%     % find central peak
%     eu_dist = sqrt(sum((peak_coords-mid_point).^2,2));  % compute euclidean distance of each peak from the centre 
%     [~,mindx] = nanmin(eu_dist);  
%     central_peak = xcorr_out(mindx).Centroid;           % closest is central peak
%     central_peak_radius = xcorr_out(mindx).EquivDiameter/2;
% else
%     central_peak_radius = 1; 
%     central_peak = [10, 10];
% end

% keyboard
% checkpoint #1: if we have less than 2 outer peaks, we revert to standard grid score
if size(peak_coords,1)<2
    out.failed_elliptical_test = 1;
    xcorr_for_grid_score = spat_xcorr;
    
else

% % find central peak
% eu_dist = sqrt(sum((peak_coords-mid_point).^2,2));  % compute euclidean distance of each peak from the centre 
% [~,mindx] = nanmin(eu_dist);  
% central_peak = xcorr_out(mindx).Centroid;           % closest is central peak
% central_peak_radius = xcorr_out(mindx).EquivDiameter/2;

% find inner ring
eu_dist = sqrt(sum((peak_coords-mid_point).^2,2));  % compute euclidean distance of each peak from the centre 
all_peaks = 1:length(eu_dist);
% outer_peaks_all = peak_coords(setdiff(all_peaks, mindx),:);         % exclude central peak

outer_peaks_dist_all = sqrt(sum((peak_coords-central_peak).^2,2));  % compute distance from central peak, sort by distance to find inner 6 peaks and furthest peak
[~, eu_dist_low_idx] = sort(outer_peaks_dist_all, 'ascend');       
peak_coords = peak_coords(eu_dist_low_idx,:);
outer_peaks_dist_all = outer_peaks_dist_all(eu_dist_low_idx);

if size(peak_coords,1)>6    % if we have more than 6 peaks, we pick closest 6
    max_peaks = 6;
else
    max_peaks = size(peak_coords,1);
end

outer_peaks = peak_coords(1:max_peaks,:);
outer_peaks_dist = outer_peaks_dist_all(1:max_peaks);
furthest_peak = outer_peaks(end-1:end,:);
% closest_peak = outer_peaks(1:2,:);
% keyboard
% generate autocorrelogram with just the inner 6 peaks
% find diameter of peaks outside inner ring and filter them out
if size(peak_coords,1)>6
    outer_xcorr(dist_mat>ceil(max(outer_peaks_dist))) = 0; 

    all_radii = cell2mat({xcorr_out.EquivDiameter}.');   
    all_radii = flip(all_radii(eu_dist_low_idx));
    outer_peaks_all_tmp = flip(peak_coords);

    for i = 1:size(peak_coords,1)-6         
        dist_mat = zeros(size(clean_xcorr)); 
        dist_mat(ceil(outer_peaks_all_tmp(i,2)),ceil(outer_peaks_all_tmp(i,1))) = 1;     
        dist_mat = bwdist(dist_mat); 
        outer_xcorr(dist_mat<ceil(all_radii(i)/2)) = 0; 
    end
end

% save variables
out.original_outer_peaks_pos = outer_peaks;
out.original_outer_peaks_dist = outer_peaks_dist;

%% 2. align major axis to horizontal axis 
% compute offset of major axis (vector going through central and furthest peak)
for i = 1:size(furthest_peak,1)
    xLen = furthest_peak(i,1)-central_peak(1);    
    yLen = furthest_peak(i,2)-central_peak(2);
    offset_maj(i) = atan2(yLen, xLen)*(180/pi);
end
[~,offset_maj_i] = min(abs(offset_maj));    % choose one that requires least rotation
offset_maj = offset_maj(offset_maj_i);

% for i = 1:size(closest_peak,1)
%     xLen = closest_peak(i,1)-central_peak(1);    
%     yLen = closest_peak(i,2)-central_peak(2);
%     offset_maj(i) = atan2(yLen, xLen)*(180/pi);
% end
% [~,offset_maj_i] = min(abs(offset_maj));
% offset_maj = offset_maj(offset_maj_i);

% keyboard

% rotate autocorrelogram 
rot_xcorr = imrotate(outer_xcorr, offset_maj, 'bilinear', 'crop');
used_nearest = 0;

% compute positions of rotated peaks
rot_xcorr_out = regionprops(logical(rot_xcorr),'Centroid', 'EquivDiameter', 'Circularity', 'Orientation'); 
rot_peak_coords = cell2mat({rot_xcorr_out.Centroid}.'); 

% sanity check - do we have the same amount of peaks after we rotate?
check_n_peaks = size(rot_peak_coords,1)==size(outer_peaks,1);                                         % 1. do we have the correct number of peaks?
if isempty(rot_peak_coords), check_x_peaks = 0;
else, check_x_peaks = sum(round(rot_peak_coords(:,2))>=9 & round(rot_peak_coords(:,2))<=11)>0; end    % 2. have peaks moved to x axis correctly?

if ~check_n_peaks || ~check_x_peaks
    % try different type of rotation
    rot_xcorr = imrotate(outer_xcorr, offset_maj, 'nearest', 'crop');
    rot_xcorr_out = regionprops(logical(rot_xcorr),'Centroid', 'EquivDiameter', 'Circularity', 'Orientation'); 
    rot_peak_coords = cell2mat({rot_xcorr_out.Centroid}.'); 
    used_nearest = 1;
end

% save variables
out.rotated_xcorr = rot_xcorr;
out.rotated_peaks_pos = rot_peak_coords;
out.rotation_offset = offset_maj;

% checkpoint #2: if rotation merges peaks, we revert to standard grid score
check_n_peaks = size(rot_peak_coords,1)==size(outer_peaks,1);                                         % 1. do we have the correct number of peaks?
if isempty(rot_peak_coords), check_x_peaks = 0;
else, check_x_peaks = sum(round(rot_peak_coords(:,2))>=9 & round(rot_peak_coords(:,2))<=11)>0; end    % 2. have peaks moved to x axis correctly?
if ~check_n_peaks || ~check_x_peaks
    out.failed_elliptical_test = 1;
    xcorr_for_grid_score = spat_xcorr;
else

%% 3. solve for minor axis
% recompute closest and furthest peaks
x_axis_peaks_i = round(rot_peak_coords(:,2))==10;
if sum(x_axis_peaks_i)==0
    x_axis_peaks_i = round(rot_peak_coords(:,2))>=9 & round(rot_peak_coords(:,2))<=11;  % two peaks on x axis are the candidates for furthest
end
rot_eu_dist = sqrt(sum((rot_peak_coords-central_peak).^2,2)); 

if size(rot_peak_coords,1)==2
    [furthest_peak_dist, furthest_i] = max(rot_eu_dist);
    furthest_peak = rot_peak_coords(furthest_i(1),:);
    closest_peak_dist = rot_eu_dist(setdiff(1:2,furthest_i(1)));
    closest_peak = rot_peak_coords(setdiff(1:2,furthest_i(1)),:);
else

if sum(x_axis_peaks_i)>2    % check whether there are more than 2 peaks aligned to x axis
rot_eu_dist_tmp = rot_eu_dist(x_axis_peaks_i);
rot_peak_coords_tmp = rot_peak_coords(x_axis_peaks_i,:);
[rot_eu_dist_tmp, eu_dist_low_idx] = sort(rot_eu_dist_tmp, 'ascend');       
rot_peak_coords_tmp = rot_peak_coords_tmp(eu_dist_low_idx,:);
furthest_peak_dist = rot_eu_dist_tmp(2);
furthest_peak = rot_peak_coords_tmp(2,:);

else

furthest_candidates = rot_peak_coords(x_axis_peaks_i,:);
[furthest_peak_dist, furthest_i] = max(rot_eu_dist(x_axis_peaks_i));
furthest_peak = furthest_candidates(furthest_i(1),:);
end

% find closest peak
% closest peak needs to be in the same quadrant as furthest to compute angle correctly
closest_candidates = rot_peak_coords(setdiff(1:max_peaks,find(x_axis_peaks_i)),:);  
closest_candidates_dist = rot_eu_dist(setdiff(1:max_peaks,find(x_axis_peaks_i)));
if furthest_peak(:,1)>central_peak(1)  
    sub_quad_i = closest_candidates(:,1)>central_peak(1); 
else
    sub_quad_i = closest_candidates(:,1)<central_peak(1); 
end
[closest_peak_dist, closest_i] = min(closest_candidates_dist(sub_quad_i)); 
sub_quad_peaks = closest_candidates(sub_quad_i,:);

try
    closest_peak = sub_quad_peaks(closest_i(1),:);
catch
    furthest_i = find(rot_peak_coords(:,1)==furthest_peak(1) & rot_peak_coords(:,2)==furthest_peak(2));
    rot_eu_dist_tmp = rot_eu_dist; rot_eu_dist_tmp(furthest_i) = [];
    rot_peak_coords_tmp = rot_peak_coords; rot_peak_coords_tmp(furthest_i,:) = [];
    [rot_eu_dist_tmp, eu_dist_low_idx] = sort(rot_eu_dist_tmp, 'ascend');       
    rot_peak_coords_tmp = rot_peak_coords_tmp(eu_dist_low_idx,:);
    closest_peak_dist = rot_eu_dist_tmp(1);
    closest_peak = rot_peak_coords_tmp(1,:);
end
end

% infer magnitude of minor axis from furthest peak (Brandon et al, 2011)
    % theta = angle between vector pointing to closest field and vector pointing to furthest field from centre
    % Rm = major axis, i.e. distance from centre to furthest peak
    % r = distance from centre to closest peak
    % rm = minor axis

P0 = central_peak;
P1 = closest_peak; 
P2 = furthest_peak;
theta = atan2(abs(det([P2-P0;P1-P0])),dot(P2-P0,P1-P0))*180/pi;

Rm = furthest_peak_dist;
r = closest_peak_dist;
rm = sqrt(sind(theta)^2 / (r^-2 -(cosd(theta)^2 / Rm^2)));

scale_ratio = rm/Rm;

% compute ellipticity and eccentricity
eccentricity = sqrt(1 -(rm/Rm)^2);
ellipticity = Rm/rm;

% save variables
out.eccentricity = eccentricity;
out.ellipticity = ellipticity;
out.Rm = Rm;
out.rm = rm;
out.r = r;

%% scale based on closest peak
% % recompute closest and furthest
% [~,x_axis_peaks_i] = sort(abs(rot_peak_coords(:,2)-10),'ascend');   % two peaks on x axis are the candidates for closest
% rot_peak_coords = rot_peak_coords(x_axis_peaks_i,:);
% 
% rot_eu_dist = sqrt(sum((rot_peak_coords-mid_point).^2,2)); 
% 
% closest_candidates = rot_peak_coords(1:2,:);
% [closest_peak_dist, closest_i] = min(rot_eu_dist(1:2));
% closest_peak = closest_candidates(closest_i(1),:);
% 
% furthest_candidates = rot_peak_coords(3:end,:);  % furthest peak needs to be in the same quadrant as furthest to compute angle correctly
% furthest_candidates_dist = rot_eu_dist(3:end);
% if closest_peak(:,1)>mid_point(1)  
%     sub_quad_i = furthest_candidates(:,1)>mid_point(1); 
% else
%     sub_quad_i = furthest_candidates(:,1)<mid_point(1); 
% end
% [furthest_peak_dist, furthest_i] = max(furthest_candidates_dist(sub_quad_i)); 
% sub_quad_peaks = furthest_candidates(sub_quad_i,:);
% furthest_peak = sub_quad_peaks(furthest_i(1),:);
% 
% 
% P0 = central_peak;
% P1 = closest_peak; 
% P2 = furthest_peak;
% theta = atan2(abs(det([P2-P0;P1-P0])),dot(P2-P0,P1-P0))*180/pi;
% 
% r = furthest_peak_dist;
% rm = closest_peak_dist;
% Rm = sqrt((-cosd(theta)^2) / ((sind(theta)^2/rm^2) -r^-2));
% scale_ratio = Rm/rm;


if ~isreal(scale_ratio) || isinf(scale_ratio) || isnan(scale_ratio) || scale_ratio==0
    out.failed_elliptical_test = 1;
    xcorr_for_grid_score = spat_xcorr;

else

%% 4. prepare autocorrelogram for grid score
final_ac = spat_xcorr;
final_ac(dist_mat<ceil(central_peak_radius)+1) = 0;                            % remove central peak
if used_nearest; final_ac = imrotate(final_ac, offset_maj, 'nearest', 'crop'); % rotate
else; final_ac = imrotate(final_ac, offset_maj, 'bilinear', 'crop'); end 

% scale x axis
scale_matrix = [scale_ratio 0 0; 0 1 0; 0 0 1];
scale_tform = affinetform2d(scale_matrix);
sameAsInput  = affineOutputView(size(final_ac),scale_tform, "BoundsStyle", "CenterOutput");
scaled_xcorr = imwarp(final_ac, scale_tform, 'OutputView', sameAsInput);

xcorr_for_grid_score = scaled_xcorr;


% % (sanity check)
% f = figure; set(gcf,'color','w');
% m = 3; n = 2;
% subplot(m,n,3)
% imagesc(flip(outer_xcorr)); title('original'); xticklabels(''); yticklabels('');
% subplot(m,n,4)
% scatter(outer_peaks(:,1),outer_peaks(:,2)); xlim([1 19]); ylim([1 19]); xline(10,'--'); yline(10,'--'); xticklabels(''); yticklabels('');
% subplot(m,n,3)
% imagesc(flip(rot_xcorr)); title('rotated'); xticklabels(''); yticklabels('');
% subplot(m,n,4)
% scatter(rot_peak_coords(:,1),rot_peak_coords(:,2)); xlim([1 19]); ylim([1 19]); xline(10,'--'); yline(10,'--'); xticklabels(''); yticklabels('');
% subplot(m,n,5)
% imagesc(flip(scaled_xcorr)); title('scaled'); xticklabels(''); yticklabels('');
% subplot(m,n,6)
% scatter(scl_peak_coords(:,1),scl_peak_coords(:,2)); xlim([1 19]); ylim([1 19]); xline(10,'--'); yline(10,'--'); xticklabels(''); yticklabels('');

% save variables
out.scale_matrix = scale_matrix;
out.scaled_xcorr = scaled_xcorr;

end
end
end

%% 5. compute grid score
rad_steps = 0.15; % amount by which to increase radii each step
angles = 30:30:150; 

min_dist  = central_peak_radius*1.5;
indiv_dist = min_dist:rad_steps:((max(size(spat_xcorr))/2)-central_peak_radius);

dist_mat = zeros(size(clean_xcorr)); 
dist_mat(ceil(central_peak(2)),ceil(central_peak(1))) = 1;     
dist_mat = bwdist(dist_mat); 

gscores_dist = nan(size(indiv_dist)); 
for e = 1:length(indiv_dist) 
    foo = xcorr_for_grid_score;
    foo((dist_mat<(indiv_dist(e)-central_peak_radius))|(dist_mat>(indiv_dist(e)+central_peak_radius))) = nan;
    % imagesc(foo)
    % pause
    curr_r = grid_score_corr(foo, angles);
    gscores_dist(e) = curr_r;
end

ell_grid_score = nanmax(gscores_dist); 
out.ell_grid_score = ell_grid_score;

end
