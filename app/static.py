SELECTION_INTRO = """
## Data Selection

For demonstration purposes, this sample application includes only two datasets.
The first one is a random sample of the pancake dataset, which is distributed along with
the package [SciKit-GStat](https://github.com/mmaelicke/scikit-gstat).
The second one is a random sample of the meuse dataset, which is distributed along with 
the R-package [gstat](https://cran.r-project.org/web/packages/gstat/index.html).

After selecting a dataset, you can choose between a number of exemplary applications.
"""

SELECTION_OUTRO = """
<hr>

If you reuse this application, please cite the following paper:

> Mälicke, Mirko, Alberto Guadagnini, and Erwin Zehe. "SciKit-GStat Uncertainty: A software extension to 
> cope with uncertain geostatistical estimates." Spatial Statistics (2023): 100737.

```bibtex
@article{MALICKE2023,
    title = {SciKit-GStat Uncertainty: A software extension to cope with uncertain geostatistical estimates},
    journal = {Spatial Statistics},
    volume = {54},
    pages = {100737},
    year = {2023},
    issn = {2211-6753},
    doi = {https://doi.org/10.1016/j.spasta.2023.100737},
    url = {https://www.sciencedirect.com/science/article/pii/S221167532300012X},
    author = {Mirko Mälicke and Alberto Guadagnini and Erwin Zehe},
    keywords = {Uncertainty, Geostatistics, Python, Variogram estimation},
}
```
"""

APP_SELECT_INTRO = """
Now, you can run either of the exemplary applications. **Uncertain empirical variograms** let's you propagate
observation uncertainties into the empirical variogram. **Multi-model parameterization** let's you fit multiple
models into the uncertainty bounds, instead of using a single model.
"""

CODE_SAMPLE = """
# -----
# This script is generated by code to reproduce the chart from above.
# You can use the snippet with appropriate attribution: https://doi.org/10.1016/j.spasta.2023.100737
# Before using this snippet, you need to install all dependencies 
# to reproduce the chart from above, you need the data sample, which is contained in the EGU repo
# link: https://github.com/hydrocode-de/egu2023/raw/main/app/data/egu.db
import requests

res = requests.get("https://github.com/hydrocode-de/egu2023/raw/main/app/data/egu.db")
with open("egu.db", "wb") as f:
    f.write(res.content)

# -----
# now you can create a API object to load the data
from skgstat_uncertainty.api import API
api = API(data_path="./", db_name="egu.db")
dataset = api.get_upload_data(id={data_id})

# -----
# estimate the variogram
from skgstat import Variogram

# build the coordinates
coords = list(zip(dataset.data['x'], dataset.data['y']))
variogram = Variogram(
    coords, 
    dataset.data['v'], 
    n_lags={bins}, 
    estimator='{estimator}',
    maxlag={maxlag}
)

"""

STD_SAMPLE = """
# -----
# estimate the uncertainty
from skgstat_uncertainty import propagation

# get the confidence interval
interval = propagation.conf_interval_from_sample_std(variogram, conf_level={conf_level} / 100)
"""

KFOLD_SAMPLE = """
# -----
# estimate the uncertainty
from skgstat_uncertainty import propagation

# get the confidence interval
interval = propagation.kfold_residual_bootstrap(variogram, k={kfold}, q=[100 - {conf_level}, {conf_level}], repititions=100)
"""

MC_SAMPLE = """
# -----
# estimate the uncertainty
from skgstat_uncertainty import propagation
import numpy as np

# The Monte Carlo method yields intermediate results. We only want the last one
res = list(propagation.mc_absolute_observation_uncertainty(variogram, sigma={sigma}, sigma_type='{sigma_method}', iterations=100))[-1]
interval = list(zip(
    np.percentile(res, q=100 - {conf_level}, axis=0),
    np.percentile(res, q={conf_level}, axis=0)
))
"""

PLOT_SAMPLE = """
# -----
# plot the results
import plotly.graph_objects as go

fig = go.Figure([
    go.Scatter(x=variogram.bins, y=variogram.experimental, name='empirical variogram', mode='markers', marker=dict(color='#8c24c7', size=10)),
    go.Scatter(x=variogram.bins, y=[b[0] for b in interval], mode='lines', line_color='#9942cb', fill=None, name='lower bounds'),
    go.Scatter(x=variogram.bins, y=[b[1] for b in interval], mode='lines', line_color='#9942cb', fill='tonexty', name='upper bounds'),
])
fig.update_layout(
    legend=dict(orientation='h')
)

# save the figure
fig.write_html("variogram.html")
fig.show()
"""