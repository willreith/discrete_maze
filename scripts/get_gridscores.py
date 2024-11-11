
from src import sac, gridscorer

area = "dACC"
cell = 64
epoch = "optionMade"
time_bins = [100]

all_g_scores = []

raw_data = sac.DataLoader(area, cell, epoch)
raw_data.load_data()

clean_data = sac.DataCleaner(raw_data)
clean_data.clean_data()

binned_data = sac.DataBinner(clean_data)
binned_data.bin_data()

smoothed_data = sac.DataSmootherOld(binned_data, smoothing_type="manual")
smoothed_data.smooth()

autocorrelograms = sac.Autocorrelograms(smoothed_data, bins=time_bins)
autocorrelograms.autocorrelate()

clean_acs = sac.CleanAutocorrelogram(autocorrelograms, threshold = 0.4)
clean_acs.clean_autocorrelograms()

peaks = sac.PeakFinder(clean_acs)
peaks.find_radius()

mask_params = sac.MaskParameters(peaks)
mask_params.make_parameters()

g_scores = []

for bin in range(len(time_bins)):

    curr_ac = autocorrelograms.autocorrelograms[..., bin]
    curr_params = mask_params.mask_parameters[bin]

    print(f"Processing bin {bin} with parameters: {curr_params}")

    g_score = gridscorer.GridScorer(curr_ac, curr_params)
    g_score.get_all_scores()
    all_g_scores.append(g_score.g_scores)

