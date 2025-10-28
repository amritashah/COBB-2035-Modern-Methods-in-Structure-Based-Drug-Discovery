import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# --- USER SETTINGS ---
# adjust these patterns to match your saved file prefixes
wt_prefix = "gtp_bound_wt_analysis"
mut_prefix = "gtp_bound_q61r_analysis"

# collect residue distance files for WT and mutant
wt_files = sorted(glob.glob(f"{wt_prefix}_res*_distances.npy"))
mut_files = sorted(glob.glob(f"{mut_prefix}_res*_distances.npy"))

# extract residue numbers using regex
def get_resid(filename):
    match = re.search(r"res(\d+)_distances\.npy", filename)
    return int(match.group(1)) if match else None

# ensure both sets have the same residues
wt_resids = [get_resid(f) for f in wt_files]
mut_resids = [get_resid(f) for f in mut_files]
common_resids = sorted(set(wt_resids) & set(mut_resids))

wt_means, mut_means = [], []

# compute mean distance for each residue
for resid in common_resids:
    wt_dist = np.load(f"{wt_prefix}_res{resid}_distances.npy")
    mut_dist = np.load(f"{mut_prefix}_res{resid}_distances.npy")
    wt_means.append(np.mean(wt_dist))
    mut_means.append(np.mean(mut_dist))

# --- Plot paired bar graph ---
x = np.arange(len(common_resids))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, wt_means, width, label='WT', color='yellowgreen', alpha=0.8)
bars2 = ax.bar(x + width/2, mut_means, width, label='Q61R', color='mediumvioletred', alpha=0.8)

ax.set_xlabel('Residue ID (selB)')
ax.set_ylabel('Mean min. distance to residue 61 (Å)')
ax.set_title('Average distances from residue 61: GTP-bound WT vs Q61R')
ax.set_xticks(x)
ax.set_xticklabels(common_resids)
ax.legend()
plt.tight_layout()
plt.savefig('gtp_wt_q61r_paired_bar.png', dpi=300, bbox_inches='tight')
plt.show()
