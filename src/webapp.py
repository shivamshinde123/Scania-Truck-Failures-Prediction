import streamlit as st
import os
import logging
from utility_function import Utility
import pandas as pd
import dill
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

make_predictions_path = params['logging_folder_paths']['make_predictions']

file_handler = logging.FileHandler(make_predictions_path)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Webapp:

    def __init__(self) -> None:
        pass

    def webapp(self):

        st.set_page_config(
            page_title='Scania Truck Failure Prediction',
            page_icon = ":truck",
            layout="centered",
            initial_sidebar_state="expanded",
        )

        st.title('Scania Truck Failure Prediction')
        st.caption('A project by Shivam Shinde')

        st.header('Metadata')
        with st.expander('Open to see the problem statement'):
            st.header('Problem Statement')

            st.markdown("""The Air Pressure System (APS) is a critical component of a heavy-duty vehicle that
                        uses compressed air to force a piston to provide pressure to the brake pads, slowing
                        the vehicle down. The benefits of using an APS instead of a hydraulic system are the
                        easy availability and long-term sustainability of natural air.
                        """)
            st.markdown("""This is a Binary Classification problem, in which the affirmative class indicates that the
                        failure was caused by a certain component of the APS, while the negative class
                        indicates that the failure was caused by something else.
                        """)

        with st.expander('Open to see trained model metrics'):
            col1, col2, col3 = st.columns(3)
            col1.metric("ROC AUC Score", "95.19 %")
            col2.metric("Precision", "75.75 %")
            col3.metric("Recall", "50 %")

        with st.expander("Open to see the plots of metrics"):
            st.image("Plots/confusion_matrix.png")
            st.image("Plots/Precision_Recall_VS_Threshold.png")
            st.image("Plots/Precision_VS_Recall.png")

        with st.expander('Please read these instructions before uploading the csv file for prediction.'):
            st.markdown("""
            Please read the following instructions before uploading the csv file for prediction.
            - The csv file should have following column names and should be in the same order: 
            class,aa_000,ab_000,ac_000,ad_000,ae_000,af_000,ag_000,ag_001,ag_002,ag_003,
            ag_004,ag_005,ag_006,ag_007,ag_008,ag_009,ah_000,ai_000,
            aj_000,ak_000,al_000,am_0,an_000,ao_000,ap_000,aq_000,ar_000,
            as_000,at_000,au_000,av_000,ax_000,ay_000,ay_001,ay_002,ay_003,ay_004,
            ay_005,ay_006,ay_007,ay_008,ay_009,az_000,az_001,az_002,az_003,
            az_004,az_005,az_006,az_007,az_008,az_009,ba_000,ba_001,ba_002,ba_003,
            ba_004,ba_005,ba_006,ba_007,ba_008,ba_009,bb_000,bc_000,bd_000, 
            be_000,bf_000,bg_000,bh_000,bi_000,bj_000,bk_000,bl_000,bm_000,bn_000,
            bo_000,bp_000,bq_000,br_000,bs_000,bt_000,bu_000,bv_000,bx_000, 
            by_000,bz_000,ca_000,cb_000,cc_000,cd_000,ce_000,cf_000,cg_000,ch_000,ci_000,
            cj_000,ck_000,cl_000,cm_000,cn_000,cn_001,cn_002,cn_003,cn_004,
            cn_005,cn_006,cn_007,cn_008,cn_009,co_000,cp_000,cq_000,cr_000,cs_000,cs_001,cs_002,
            cs_003,cs_004,cs_005,cs_006,cs_007,cs_008,cs_009,ct_000,cu_000,
            cv_000,cx_000,cy_000,cz_000,da_000,db_000,dc_000,dd_000,de_000,df_000,dg_000,dh_000,
            di_000,dj_000,dk_000,dl_000,dm_000,dn_000,do_000,dp_000,dq_000,dr_000,
            ds_000,dt_000,du_000,dv_000,dx_000,dy_000,dz_000,ea_000,eb_000,ec_00,
            ed_000,ee_000,ee_001,ee_002,
            ee_003,ee_004,ee_005,ee_006,ee_007,ee_008,ee_009,ef_000,eg_000
            - All the value should be integer or float except for the class column which has text values.
            """)

        st.header("Make Prediction")
        st.markdown("Upload a csv file adhering to the above mentioned instructions for the prediction:")
        file = st.file_uploader('Upload a CSV file', type={'csv', 'txt'})

        if file is not None:
            input = pd.read_csv(file)

            input.drop(columns=['ab_000', 'ad_000', 'bk_000', 'bl_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000', 'br_000', 'cf_000', 'cg_000', 'ch_000', 'co_000', 'cr_000',
                        'ct_000', 'cu_000', 'cv_000', 'cx_000', 'cy_000', 'cz_000', 'da_000', 'db_000', 'dc_000', 'class'],
               axis=1, inplace=True)

            preprocess_pipe_folderpath = params['model']['preprocess_pipe_folderpath']
            preprocess_pipe_filename = params['model']['preprocess_pipe_filename']

            preprocess_pipe_path = os.path.join(
                preprocess_pipe_folderpath, preprocess_pipe_filename)
            with open(preprocess_pipe_path, 'rb') as f:
                preprocess_pipe = dill.load(f)

            data_input = preprocess_pipe.transform(input)
        
        if st.button("Make Prediction"):
            with st.spinner('Please wait'):
                model_foldername = params['model']['model_foldername']
                model_name = params['model']['model_name']

                with open(os.path.join(model_foldername, model_name), 'rb') as f:
                    model = dill.load(f)

                predictions = model.predict(data_input)

                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(pd.DataFrame(predictions))

                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )

        

if __name__ == "__main__":
    wa = Webapp()
    wa.webapp()
