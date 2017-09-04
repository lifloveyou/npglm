import numpy as np
import matplotlib.pyplot as plt
from codes.models import get_dist_rnd, generate_data
from codes.models.NpGlm import NpGlm
from codes.models.tvregdiff import TVRegDiff

np.random.seed(0)
d = 10
n = 1000
dist = 'gom'
dist_rnd = get_dist_rnd(dist)
w = np.random.randn(d + 1, 1)
X, T = generate_data(w, n, dist_rnd)
Y = np.ones((n,), dtype=bool)
model = NpGlm()
model.fit(X, Y, T)
limit = 1000
t = model.t[:limit].ravel()
H = model.H[:limit]
plt.plot(t, H)
h = TVRegDiff(H, 500, 10, plotflag=False, scale='large')
plt.plot(t, h[:limit])
plt.show()
