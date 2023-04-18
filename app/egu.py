import os

import streamlit as st
from skgstat_uncertainty.api import API
from streamlit_card_select import card_select

from static import SELECTION_INTRO


# data path
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

# for demonstration, we always use the same data -> random sample of pancake ID=6 or meuse ID= 3
PAN_ID = 6
MEUSE_ID = 3



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
    select = card_select(options)
    if select is not None:
        st.session_state.data_id = select
        st.experimental_rerun()


def main(api: API) -> None:
    st.title('EGU 2023')


def run(data_path=DATA_PATH, db_name='egu.db'):
        api = API(data_path=data_path, db_name=db_name)
        main(api)

        if 'data_id' not in st.session_state:
            data_selection(api)
        else:
            st.write("You selected data ID=", st.session_state.data_id)


if __name__ == '__main__':
    # set title and logo
    st.set_page_config(page_title='EGU 2023', layout='wide')
    st.sidebar.image("https://firebasestorage.googleapis.com/v0/b/hydrocode-website.appspot.com/o/public%2Fhydrocode_brand.png?alt=media")
    
    # run the main app
    import fire
    fire.Fire(run)

    # add extra links
    with st.sidebar.expander("More info", expanded=True):
        # place a couple of links
        st.markdown("""<a href="https://hydrocode.de" target="_blank">hydrocode GmbH</a>""", unsafe_allow_html=True)
