function out = grid_score_corr(autocorr, angles)

 % keyboard
    corrs = nan(length(angles), 1);
    for e = 1:length(angles)
        autocorr_rot = imrotate(autocorr,angles(e),'bilinear','crop');
        tmp = corrcoef(autocorr(:), autocorr_rot(:),'rows','pairwise');
        corrs(e) = tmp(1, 2);
    end

    g = nanmin([corrs(2),corrs(4)]) - nanmax([corrs(1),corrs(3),corrs(5)]);

    % "minimum difference between between any of the elements in the first group and any of the elements in the second." (langston 2010)
    % --- doesn't this mean min(r(60)-r(30), r(120)-r(30), r(60)-r(90), ...)

    out = g;
end