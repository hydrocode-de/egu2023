from typing import List, Union
import os
import hashlib

import streamlit as st
import numpy as np
from skgstat import Variogram
from skgstat_uncertainty.api import API
from skgstat_uncertainty.components.utils import ESTIMATORS, MODELS, CONF_METHODS
from skgstat_uncertainty.processor import propagation
from streamlit_card_select import card_select
import plotly.graph_objects as go

from static import SELECTION_INTRO, CODE_SAMPLE, STD_SAMPLE, KFOLD_SAMPLE, MC_SAMPLE, PLOT_SAMPLE


# data path
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

# for demonstration, we always use the same data -> random sample of pancake ID=6 or meuse ID= 3
PAN_ID = 6
MEUSE_ID = 3

# filter out the dowd estimator
FILT_ESTIMATORS = {k: v for k, v in ESTIMATORS.items() if k not in ('genton', 'entropy')}

# filter out confidence estimation methods
FILT_CONF_METHODS = {k: v for k, v in CONF_METHODS.items() if k != 'residual'}



def estimate_variogram(api: API) -> None:
    # create an empty container for the plot
    plot_area = st.empty()

    # get the dataset
    data = api.get_upload_data(id=st.session_state.data_id)

    # render variogram estimation params in three columns
    left, center = st.columns(2)

    # variogram options
    left.markdown('#### empirical variogram')
    left.number_input('Number of bins', min_value=1, max_value=25, value=15, step=1, key='bins')
    left.selectbox('Estimator', options=FILT_ESTIMATORS, index=0, format_func=lambda k: FILT_ESTIMATORS.get(k), key='estimator')
    OPT = {'mean': 'Mean separating distance', 'median': 'Median separating distance', 'none': 'No maximum lag distance'}
    left.selectbox('Maximum lag distance', options=OPT, index=0, format_func=lambda k: OPT.get(k), key='maxlag')

    # uncertainty estimation options
    center.markdown('#### uncertainty estimation')
    center.selectbox('Uncertainty estimation method', options=FILT_CONF_METHODS, index=0, format_func=lambda k: CONF_METHODS.get(k), key='conf_method')
    
    # every selection method has a confidence level
    center.slider('symmetric quantiles', min_value=50, max_value=99, value=95, step=1, key='conf_level')
    
    # switch the confidence estimation method
    if st.session_state.conf_method == 'kfold':    
        center.slider('Number of folds (k)', min_value=2, max_value=10, value=7, step=1, key='kfold')
        center.info('In this demo app, the repetitions are set to only 100.')
    elif st.session_state.conf_method == 'absolute':
        OPT = {'sem': 'Standard Error of the Mean', 'std': 'Standard Deviation of observation', 'precision': 'Absolute precision of observation'}
        center.selectbox('Observation uncertainty type', options=OPT, index=0, format_func=lambda k: OPT.get(k), key='sigma_method')
        center.slider('Uncertainty', min_value=0.0, max_value=10.0, value=0.1, step=0.01, key='sigma')
        center.info('In this demo app, the repetitions are set to only 100.')
    
    # now the variogram is needed
    variogram = cache_variogram(
        {k:v for k,v in data.data.items() if k in ('x', 'y', 'v')}, 
        bins=st.session_state.bins,
        estimator=st.session_state.estimator,
        maxlag=st.session_state.maxlag if st.session_state.maxlag != 'none' else None
    )

    # vario_md5 this is needed to correctly refresh the cache of uncertainty estimation as Variogram is not hashable
    vario_md5 = hashlib.md5(f"{st.session_state.data_id};{st.session_state.bins};{st.session_state.estimator};{st.session_state.maxlag}".encode('utf-8')).hexdigest()


    # run the estimation
    interval = estimate_uncertainty(
        vario_md5,
        variogram,
        method=st.session_state.conf_method,
        conf_level=st.session_state.conf_level,
        kfold=st.session_state.get('kfold'),
        sigma_method=st.session_state.get('sigma_method'),
        sigma=st.session_state.get('sigma')
    )

    # now fill the plot
    fig = go.Figure([
        go.Scatter(x=variogram.bins, y=variogram.experimental, name='empirical variogram', mode='markers', marker=dict(color='#8c24c7', size=10)),
        go.Scatter(x=variogram.bins, y=[b[0] for b in interval], mode='lines', line_color='#9942cb', fill=None, name='lower bounds'),
        go.Scatter(x=variogram.bins, y=[b[1] for b in interval], mode='lines', line_color='#9942cb', fill='tonexty', name='upper bounds'),
    ])
    fig.update_layout(
        legend=dict(orientation='h')
    )
    plot_area.plotly_chart(fig, use_container_width=True)

    # finally plot the code
    with st.expander('CODE SAMPLE', expanded=True):
        st.markdown('#### Code')
        code = CODE_SAMPLE.format(
            data_id=st.session_state.data_id,
            bins=st.session_state.bins,
            estimator=st.session_state.estimator,
            maxlag=f"'{st.session_state.maxlag}'" if st.session_state.maxlag != 'none' else 'None',
        )
        
        # uncertainty estimation sample
        if st.session_state.conf_method == 'std':
            code += STD_SAMPLE.format(conf_level=st.session_state.conf_level)
        elif st.session_state.conf_method == 'kfold':
            code += KFOLD_SAMPLE.format(conf_level=st.session_state.conf_level, kfold=st.session_state.kfold)
        elif st.session_state.conf_method == 'absolute':
            code += MC_SAMPLE.format(conf_level=st.session_state.conf_level, sigma_method=st.session_state.sigma_method, sigma=st.session_state.sigma)
        
        # plot sample
        code += PLOT_SAMPLE
        
        # print
        st.code(code)


@st.cache_data
def cache_variogram(data: dict, bins: int, estimator: str, maxlag: Union[str, float]) -> Variogram:
    # build the coordinates
    coordinates = np.asarray(list(zip(data['x'], data['y'])))
    values = np.asarray(data['v'])
    return Variogram(coordinates, values, n_lags=bins, estimator=estimator, maxlag=maxlag)


@st.cache_data
def estimate_uncertainty(vario_md5, _variogram: Variogram, method: str, conf_level: float, kfold: int = None, sigma_method: str = None, sigma: float = None) -> List[List[float]]:
    if method == 'std':
        interval = propagation.conf_interval_from_sample_std(_variogram, conf_level=conf_level / 100)
    elif method == 'kfold':
        interval = propagation.kfold_residual_bootstrap(_variogram, k=kfold, q=[100-conf_level, conf_level], repititions=100)
    elif method == 'absolute':
        res = propagation.mc_absolute_observation_uncertainty(_variogram, sigma=sigma, sigma_type=sigma_method, iterations=100)
        
        # the mc returns intermediate results. We only want the last one
        result = list(res)[-1]
        interval = list(zip(
            np.percentile(result, q=100-conf_level, axis=0),
            np.percentile(result, q=conf_level, axis=0)
        ))
    else:
        st.error(f'Unknown method "{method}". Somthing is wrong.')
        st.stop()

    return interval


def data_selection(api: API) -> None:
    st.markdown(SELECTION_INTRO)
    
    # load both datasets
    pancake = api.get_upload_data(id=6)
    meuse = api.get_upload_data(id=3)

    # build the options for the card_select
    options = [
        dict(option=6, title="Pancake", description=pancake.data.get('description', 'no description available'), image=pancake.data['thumbnail']),
        dict(option=3, title="Meuse", description=meuse.data.get('description', 'no description available'), image=meuse.data['thumbnail'])
    ]

    # get an option from the user
    select = card_select(options, md="6", xs="12", lg="6")
    if select is not None:
        st.session_state.data_id = select
        st.experimental_rerun()


def subapp_selection() -> None:
     pass


def main(api: API) -> None:
    st.title('EGU 2023')

    if 'data_id' not in st.session_state:
        data_selection(api=api)
    else:
        estimate_variogram(api=api)


def run(data_path=DATA_PATH, db_name='egu.db'):
        api = API(data_path=data_path, db_name=db_name)
        main(api)


if __name__ == '__main__':
    # set title and logo
    st.set_page_config(page_title='EGU 2023', layout='wide')
    st.sidebar.image("https://firebasestorage.googleapis.com/v0/b/hydrocode-website.appspot.com/o/public%2Fhydrocode_brand.png?alt=media")
    
    # run the main app
    import fire
    fire.Fire(run)

    # add extra links
    with st.sidebar.expander("More info", expanded=True):
        # on-site info
        st.markdown("visit the on-site Poster on Wednesday 26.04 **10:45-12:30** in **Hall A A.101**")
        # place a couple of links
        st.markdown("""<a href="https://doi.org/10.5194/egusphere-egu23-6683" target="_blank">EGU 2023 Abstract</a>""", unsafe_allow_html=True)
        st.markdown("""<a href="https://github.com/hydrocode-de/egu2023" target="_blank">GitHub repo</a>""", unsafe_allow_html=True)
        st.markdown("""<a href="https://hydrocode.de" target="_blank">hydrocode GmbH</a>""", unsafe_allow_html=True)
