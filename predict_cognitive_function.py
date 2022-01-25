import os
import pandas as pd
import numpy as np
import nibabel as nb
from nilearn.image import load_img
from nilearn.input_data import NiftiMasker
from nilearn.input_data import NiftiLabelsMasker
from sklearn.externals import joblib

def predict(output='output.csv',
         measure='CogTotalComp',
         gray_matter=None,
         white_matter=None,
         rfMRI_basc_correlation_pe_LR=None,
         rfMRI_basc_correlation_pe_RL=None,
         rfMRI_basc_partial_correlation_pe_LR=None,
         rfMRI_basc_partial_correlation_pe_RL=None,
         rfMRI_basc_tangent_pe_LR=None,
         rfMRI_basc_tangent_pe_RL=None,
         rfMRI_REST1_hp2000_clean_s4_0_000_to_0_250_ALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_ALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_ALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_ALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_ALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_fALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_fALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_fALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_fALFF=None,
         rfMRI_REST1_hp2000_clean_s4_0_6_LFCD=None,
         rfMRI_REST1_hp2000_clean_s4_1_3dDegreeCentrality=None,
         rfMRI_REST1_hp2000_clean_s4_ReHo=None,
         tfMRI_WM_2BK_s4=None,
         tfMRI_WM_0BK_s4=None,
         tfMRI_WM_2BK_minus_0BK_s4=None,
         tfMRI_WM_BODY_s4=None,
         tfMRI_WM_FACE_s4=None,
         tfMRI_WM_PLACE_s4=None,
         tfMRI_WM_TOOL_s4=None,
         tfMRI_RELATIONAL_REL_s4=None,
         tfMRI_RELATIONAL_MATCH_s4=None,
         tfMRI_RELATIONAL_REL_minus_MATCH_s4=None,
         tfMRI_EMOTION_FACES_s4=None,
         tfMRI_EMOTION_SHAPES_s4=None,
         tfMRI_EMOTION_FACES_minus_SHAPES_s4=None,
         tfMRI_LANGUAGE_STORY_s4=None,
         tfMRI_LANGUAGE_MATH_s4=None,
         tfMRI_LANGUAGE_STORY_minus_MATH_s4=None,
         tfMRI_GAMBLING_REWARD_s4=None,
         tfMRI_GAMBLING_PUNISH_s4=None,
         tfMRI_SOCIAL_TOM_s4=None,
         tfMRI_SOCIAL_RANDOM_s4=None,
         tfMRI_SOCIAL_TOM_minus_RANDOM_s4=None,
         winsorizing='yes',
         scaling='yes',
         pca='yes',
         intercept='yes'):
    
    arguments = locals()
    
#    return arguments
    
    features=['gray_matter',
           'white_matter',
           'rfMRI_basc_correlation',
           'rfMRI_basc_partial_correlation',
           'rfMRI_basc_tangent',
           'rfMRI_REST1_hp2000_clean_s4_0_000_to_0_250_ALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_ALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_ALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_ALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_ALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_fALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_fALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_fALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_fALFF',
           'rfMRI_REST1_hp2000_clean_s4_0_6_LFCD',
           'rfMRI_REST1_hp2000_clean_s4_1_3dDegreeCentrality',
           'rfMRI_REST1_hp2000_clean_s4_ReHo',
           'tfMRI_WM_2BK_s4',
           'tfMRI_WM_0BK_s4',
           'tfMRI_WM_2BK_minus_0BK_s4',
           'tfMRI_WM_BODY_s4',
           'tfMRI_WM_FACE_s4',
           'tfMRI_WM_PLACE_s4',
           'tfMRI_WM_TOOL_s4',
           'tfMRI_RELATIONAL_REL_s4',
           'tfMRI_RELATIONAL_MATCH_s4',
           'tfMRI_RELATIONAL_REL_minus_MATCH_s4',
           'tfMRI_EMOTION_FACES_s4',
           'tfMRI_EMOTION_SHAPES_s4',
           'tfMRI_EMOTION_FACES_minus_SHAPES_s4',
           'tfMRI_LANGUAGE_STORY_s4',
           'tfMRI_LANGUAGE_MATH_s4',
           'tfMRI_LANGUAGE_STORY_minus_MATH_s4',
           'tfMRI_GAMBLING_REWARD_s4',
           'tfMRI_GAMBLING_PUNISH_s4',
           'tfMRI_SOCIAL_TOM_s4',
           'tfMRI_SOCIAL_RANDOM_s4',
           'tfMRI_SOCIAL_TOM_minus_RANDOM_s4']
    
    results=pd.DataFrame()
    
    '''        
    #######################################################################
    #######################################################################
    #######################################################################
    # First-level pipelines
    #######################################################################
    #######################################################################
    #######################################################################
    '''
    
    for feature in features:
        print(feature)
        
        y_pred = np.nan
        pe_LR_filename = None
        pe_RL_filename = None
        
        if 'basc' in feature:
            
            if 'basc_correlation' in feature:
                pe_LR_filename = arguments.get('rfMRI_basc_correlation_pe_LR')
                pe_RL_filename = arguments.get('rfMRI_basc_correlation_pe_RL')
            
            elif 'basc_partial_correlation' in feature:
                pe_LR_filename = arguments.get('rfMRI_basc_partial_correlation_pe_LR')
                pe_RL_filename = arguments.get('rfMRI_basc_partial_correlation_pe_RL')
                
            elif 'basc_tangent' in feature:
                pe_LR_filename = arguments.get('rfMRI_basc_tangent_pe_LR')
                pe_RL_filename = arguments.get('rfMRI_basc_tangent_pe_RL')
            
            if pe_LR_filename != None or pe_RL_filename != None:
                                    
                try:
                    
                    print("Loading basc atlas for %s" % (feature))
                    
                    atlas_filename = './atlases/basc_multiscale_2015/template_cambridge_basc_multiscale_nii_asym/template_cambridge_basc_multiscale_asym_scale122.nii.gz'
                    masker = NiftiLabelsMasker(atlas_filename, resampling_target='data', smoothing_fwhm=None, standardize=False,
                                               detrend=True, low_pass=0.1, high_pass=0.008, memory_level=0, memory=None)
                    
                    print("Loading first-level pipeline for %s" % (feature))
                    pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_PCA_LinearRegression_l2_regression_analysis_first_level_pipeline_and_results.sav')
                                                                          
                    if pe_LR_filename != None and pe_RL_filename == None:
                        print("Loadingfirst-level  %s for %s" % (pe_LR_filename, feature))
                        pe_LR_t_r = nb.load(pe_LR_filename).header.get_zooms()[3]
                        masker.t_r = pe_LR_t_r
                        time_series_pe_LR = masker.fit_transform(pe_LR_filename)
                        conn_measure_pe_LR = pipeline['conn_measure_LR']
                        data = conn_measure_pe_LR.transform([time_series_pe_LR])
                        
                        if feature == 'rfMRI_basc_correlation' or feature == 'rfMRI_basc_partial_correlation':
                            data = np.arctanh(data)
                        
                    elif pe_LR_filename == None and pe_RL_filename != None:
                        print("Loading first-level %s for %s" % (pe_RL_filename, feature))
                        pe_RL_t_r = nb.load(pe_RL_filename).header.get_zooms()[3]
                        masker.t_r = pe_RL_t_r
                        time_series_pe_RL = masker.fit_transform( pe_RL_filename)
                        conn_measure_pe_RL = pipeline['conn_measure_RL']
                        data = conn_measure_pe_RL.transform([time_series_pe_RL])
                        
                        if feature == 'rfMRI_basc_correlation' or feature == 'rfMRI_basc_partial_correlation':
                            data = np.arctanh(data)
                        
                    elif pe_LR_filename != None and pe_RL_filename != None:
                        print("Loading first-level %s for %s" % (pe_LR_filename, feature))
                        pe_LR_t_r = nb.load(pe_LR_filename).header.get_zooms()[3]
                        masker.t_r = pe_LR_t_r
                        time_series_pe_LR = masker.fit_transform(pe_LR_filename)
                        conn_measure_pe_LR = pipeline['conn_measure_LR']
                        data_LR = conn_measure_pe_LR.transform([time_series_pe_LR])
                        
                        print("Loading first-level %s for %s" % (pe_RL_filename, feature))                            
                        pe_RL_t_r = nb.load(pe_RL_filename).header.get_zooms()[3]
                        masker.t_r = pe_RL_t_r
                        time_series_pe_RL = masker.fit_transform(pe_RL_filename)
                        conn_measure_pe_RL = pipeline['conn_measure_RL']
                        data_RL = conn_measure_pe_RL.transform([time_series_pe_RL])
                        
                        if feature == 'rfMRI_basc_correlation' or feature == 'rfMRI_basc_partial_correlation':
                            data_LR = np.arctanh(data_LR)
                            data_RL = np.arctanh(data_RL)
                  
                        data = (data_LR + data_RL) / 2                            
              
                except ValueError:
                    print("Error loading first-level %s data for %s! Exiting." % (arguments.get(feature), feature))      
                    
        elif arguments.get(feature) != None:
            print(arguments.get(feature))
            
            try:
                
                if feature == 'gray_matter':
                    print("Loading first-level %s for %s" % (arguments.get(feature), feature))
                    data = pd.read_csv(arguments.get(feature), header=0)
                    data = data.values
                    
                elif feature == 'white_matter':
                    print("Loading first-level %s for %s" % (arguments.get(feature), feature))
                    white_matter_mask = load_img('./atlases/white_matter_mask.nii.gz')
                    masker = NiftiMasker(mask_img=white_matter_mask)
                    data = masker.fit_transform(arguments.get(feature))
                
                elif 'tfMRI' in feature or 'ALFF' in feature or 'LFCD' in feature or '3dDegreeCentrality' in feature or 'ReHo' in feature:
                    print("Loading first-level %s data for %s" % (arguments.get(feature), feature))
                    gray_matter_mask = load_img('./atlases/gray_matter_mask.nii.gz')
                    masker = NiftiMasker(mask_img=gray_matter_mask)
                    data = masker.fit_transform(arguments.get(feature))
                    
            except ValueError:
                print("Error loading first-level %s data for %s! Exiting." % (arguments.get(feature), feature))
                
            try:  
                
                if 'basc' not in feature and feature != 'gray_matter':                    
                    print("Loading first-level pipeline for %s" % (feature))
                    pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_PCA_LinearRegression_l2_regression_analysis_first_level_pipeline_and_results.sav')
                    
                elif feature == 'gray_matter':                    
                    print("Loading first-level pipeline for %s" % (feature))
                    pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_no_feature_selection_LinearRegression_l2_regression_analysis_first_level_pipeline_and_results.sav')
            
            except ValueError:
                print("Error loading first-level %s pipeline for %s! Exiting." % (arguments.get(feature), feature))
                
        if arguments.get(feature) != None or pe_LR_filename != None or pe_RL_filename != None:
            
            try:
        
                if winsorizing == 'yes':
                    print('Winsorizing')
                    lower = pipeline['lower']
                    upper = pipeline['upper']
                    data = np.clip(data, lower, upper)
                    
                if scaling == 'yes':
                    print('Using scaling')
                    sc_X_train = pipeline['sc_X_train']    
                    data = sc_X_train.transform(data)
                
                if pca == 'yes' and feature != 'gray_matter':
                    print('Using PCA')
                    feature_select = pipeline['feature_select']
                    data = feature_select.transform(data)
    
                regressor=pipeline['regressor']
                
                if pca != 'yes' and feature != 'gray_matter':
                    feature_select = pipeline['feature_select']
                    regressor.coef_ = feature_select.inverse_transform(regressor.coef_)
                    
                if intercept != 'yes':
                    print('No intercept: Setting intercept to 0')
                    regressor.intercept_ = np.array([0], dtype='float32')
                
                y_pred = regressor.predict(data)
                
                sc_y_train=pipeline['sc_y_train']
                y_pred = sc_y_train.inverse_transform(y_pred)
                
            except ValueError:
                print("Error with first-level prediction for %s! Exiting." % (arguments.get(feature), feature))
                
        else:
            print("Skipping %s" % (feature))
            
        if np.isnan(y_pred):
            results[feature] = y_pred
        else:
            results[feature] = y_pred[0]
    
    '''        
    #######################################################################
    #######################################################################
    #######################################################################
    # Second-level pipelines
    #######################################################################
    #######################################################################
    #######################################################################
    '''
    
    features=['fc', 'rs', 'WM', 'CATEGORY', 'RELATIONAL', 'EMOTION', 'LANGUAGE', 'GAMBLING', 'SOCIAL']
  
    for feature in features:
        
        y_pred = np.nan
        
        #######################################################################
        # Functional Connectivity second-level pipeline
        #######################################################################
        
        if feature == 'fc' and (pd.notnull(results['rfMRI_basc_correlation'][0])
            and pd.notnull(results['rfMRI_basc_partial_correlation'][0])
            and pd.notnull(results['rfMRI_basc_tangent'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['rfMRI_basc_correlation'][0],
                             results['rfMRI_basc_partial_correlation'][0],
                             results['rfMRI_basc_tangent'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
            
            y_pred = regressor.predict(data.reshape(1,-1))             
        
        #######################################################################
        # Frequency Domain and Graph Measures
        #######################################################################
        
        elif feature == 'rs' and(pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_000_to_0_250_ALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_ALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_ALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_ALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_ALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_fALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_fALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_fALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_fALFF'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_0_6_LFCD'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_1_3dDegreeCentrality'][0])
            and pd.notnull(results['rfMRI_REST1_hp2000_clean_s4_ReHo'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['rfMRI_REST1_hp2000_clean_s4_0_000_to_0_250_ALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_ALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_ALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_ALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_ALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_010_to_0_027_fALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_027_to_0_073_fALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_073_to_0_167_fALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_167_to_0_250_fALFF'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_0_6_LFCD'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_1_3dDegreeCentrality'][0],
                            results['rfMRI_REST1_hp2000_clean_s4_ReHo'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
            
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # Working Memory second-level pipeline
        #######################################################################
            
        elif feature == 'WM' and (pd.notnull(results['tfMRI_WM_0BK_s4'][0])
            and pd.notnull(results['tfMRI_WM_2BK_s4'][0])
            and pd.notnull(results['tfMRI_WM_2BK_minus_0BK_s4'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['tfMRI_WM_0BK_s4'][0],
                             results['tfMRI_WM_2BK_s4'][0],
                             results['tfMRI_WM_2BK_minus_0BK_s4'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))                     
        
        #######################################################################
        # Category-Specific Representations second-level pipeline
        #######################################################################
            
        elif feature == 'CATEGORY' and (pd.notnull(results['tfMRI_WM_TOOL_s4'][0])
            and pd.notnull(results['tfMRI_WM_PLACE_s4'][0])
            and pd.notnull(results['tfMRI_WM_FACE_s4'][0])
            and pd.notnull(results['tfMRI_WM_BODY_s4'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['tfMRI_WM_TOOL_s4'][0],
                             results['tfMRI_WM_PLACE_s4'][0],
                             results['tfMRI_WM_FACE_s4'][0],
                             results['tfMRI_WM_BODY_s4'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # Relational second-level pipeline
        #######################################################################
        
        elif feature == 'RELATIONAL' and (pd.notnull(results['tfMRI_RELATIONAL_MATCH_s4'][0])
            and pd.notnull(results['tfMRI_RELATIONAL_REL_s4'][0])
            and pd.notnull(results['tfMRI_RELATIONAL_REL_minus_MATCH_s4'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['tfMRI_RELATIONAL_MATCH_s4'][0],
                             results['tfMRI_RELATIONAL_REL_s4'][0],
                             results['tfMRI_RELATIONAL_REL_minus_MATCH_s4'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))

        #######################################################################
        # Emotion second-level pipeline
        #######################################################################
        
        elif feature == 'EMOTION' and (pd.notnull(results['tfMRI_EMOTION_FACES_s4'][0])
            and pd.notnull(results['tfMRI_EMOTION_SHAPES_s4'][0])
            and pd.notnull(results['tfMRI_EMOTION_FACES_minus_SHAPES_s4'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['tfMRI_EMOTION_FACES_s4'][0],
                             results['tfMRI_EMOTION_SHAPES_s4'][0],
                             results['tfMRI_EMOTION_FACES_minus_SHAPES_s4'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # Language second-level pipeline
        #######################################################################
        
        elif feature == 'LANGUAGE' and (pd.notnull(results['tfMRI_LANGUAGE_MATH_s4'][0])
            and pd.notnull(results['tfMRI_LANGUAGE_STORY_s4'][0])
            and pd.notnull(results['tfMRI_LANGUAGE_STORY_minus_MATH_s4'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['tfMRI_LANGUAGE_MATH_s4'][0],
                             results['tfMRI_LANGUAGE_STORY_s4'][0],
                             results['tfMRI_LANGUAGE_STORY_minus_MATH_s4'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # Gambling second-level pipeline
        #######################################################################
        
        elif feature == 'GAMBLING' and (pd.notnull(results['tfMRI_GAMBLING_PUNISH_s4'][0])
            and pd.notnull(results['tfMRI_GAMBLING_REWARD_s4'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['tfMRI_GAMBLING_PUNISH_s4'][0],
                             results['tfMRI_GAMBLING_REWARD_s4'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
        
        #######################################################################
        # Social second-level pipeline
        #######################################################################
        
        elif feature == 'SOCIAL' and (pd.notnull(results['tfMRI_SOCIAL_RANDOM_s4'][0])
            and pd.notnull(results['tfMRI_SOCIAL_TOM_s4'][0])
            and pd.notnull(results['tfMRI_SOCIAL_TOM_minus_RANDOM_s4'][0])):
            
            print("Loading second-level pipeline for %s" % (feature))
            
            data=np.asarray([results['tfMRI_SOCIAL_RANDOM_s4'][0],
                             results['tfMRI_SOCIAL_TOM_s4'][0],
                             results['tfMRI_SOCIAL_TOM_minus_RANDOM_s4'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_second_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
        
        else:
            print("Skipping second-level pipeline for %s" % (feature))
      
        if np.isnan(y_pred):
            results[feature] = y_pred
        else:
            results[feature] = y_pred[0]
            
    '''        
    #######################################################################
    #######################################################################
    #######################################################################
    # Third-level pipelines
    #######################################################################
    #######################################################################
    #######################################################################
    '''
    
    features=['gray_matter_white_matter_fc_rs_WM_CATEGORY_RELATIONAL_EMOTION_LANGUAGE_GAMBLING_SOCIAL',
              'gray_matter_white_matter_fc_rs_CATEGORY_RELATIONAL_EMOTION_LANGUAGE_GAMBLING_SOCIAL',
              'gray_matter_white_matter',
              'gray_matter_white_matter_fc_rs',
              'gray_matter_white_matter_fc_rs_EMOTION',
              'gray_matter_white_matter_fc_rs_WM']
  
    for feature in features:
        
        y_pred = np.nan
        
        #######################################################################
        # All feature sets
        #######################################################################
        
        if feature == 'gray_matter_white_matter_fc_rs_WM_CATEGORY_RELATIONAL_EMOTION_LANGUAGE_GAMBLING_SOCIAL' and (pd.notnull(results['gray_matter'][0])
            and pd.notnull(results['white_matter'][0])
            and pd.notnull(results['fc'][0])
            and pd.notnull(results['rs'][0])
            and pd.notnull(results['WM'][0])
            and pd.notnull(results['CATEGORY'][0])
            and pd.notnull(results['RELATIONAL'][0])
            and pd.notnull(results['EMOTION'][0])
            and pd.notnull(results['LANGUAGE'][0])
            and pd.notnull(results['GAMBLING'][0])
            and pd.notnull(results['SOCIAL'][0])):
            
            print("Loading third-level pipeline for %s" % (feature))
            
            data=np.asarray([results['gray_matter'][0],
                            results['white_matter'][0],
                            results['fc'][0],
                            results['rs'][0],
                            results['WM'][0],
                            results['CATEGORY'][0],
                            results['RELATIONAL'][0],
                            results['EMOTION'][0],
                            results['LANGUAGE'][0],
                            results['GAMBLING'][0],
                            results['SOCIAL'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_third_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # All Features Sets Except for Working Memory
        #######################################################################
        
        if feature == 'gray_matter_white_matter_fc_rs_CATEGORY_RELATIONAL_EMOTION_LANGUAGE_GAMBLING_SOCIAL' and (pd.notnull(results['gray_matter'][0])
            and pd.notnull(results['white_matter'][0])
            and pd.notnull(results['fc'][0])
            and pd.notnull(results['rs'][0])
            and pd.notnull(results['CATEGORY'][0])
            and pd.notnull(results['RELATIONAL'][0])
            and pd.notnull(results['EMOTION'][0])
            and pd.notnull(results['LANGUAGE'][0])
            and pd.notnull(results['GAMBLING'][0])
            and pd.notnull(results['SOCIAL'][0])):
            
            print("Loading third-level pipeline for %s" % (feature))
            
            data=np.asarray([results['gray_matter'][0],
                            results['white_matter'][0],
                            results['fc'][0],
                            results['rs'][0],
                            results['CATEGORY'][0],
                            results['RELATIONAL'][0],
                            results['EMOTION'][0],
                            results['LANGUAGE'][0],
                            results['GAMBLING'][0],
                            results['SOCIAL'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_third_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))        

        #######################################################################
        # Gray Matter and White Matter
        #######################################################################
        
        if feature == 'gray_matter_white_matter' and (pd.notnull(results['gray_matter'][0])
            and pd.notnull(results['white_matter'][0])):
            
            print("Loading third-level pipeline for %s" % (feature))
            
            data=np.asarray([results['gray_matter'][0],
                            results['white_matter'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_third_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # Gray Matter, White Matter, and Resting State
        #######################################################################
        
        if feature == 'gray_matter_white_matter_fc_rs' and (pd.notnull(results['gray_matter'][0])
            and pd.notnull(results['white_matter'][0])
            and pd.notnull(results['fc'][0])
            and pd.notnull(results['rs'][0])):
            
            print("Loading third-level pipeline for %s" % (feature))
            
            data=np.asarray([results['gray_matter'][0],
                            results['white_matter'][0],
                            results['fc'][0],
                            results['rs'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_third_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # Gray Matter, White Matter, Resting State, and Emotion Processing
        #######################################################################
        
        if feature == 'gray_matter_white_matter_fc_rs_EMOTION' and (pd.notnull(results['gray_matter'][0])
            and pd.notnull(results['white_matter'][0])
            and pd.notnull(results['fc'][0])
            and pd.notnull(results['rs'][0])
            and pd.notnull(results['EMOTION'][0])):
            
            print("Loading third-level pipeline for %s" % (feature))
            
            data=np.asarray([results['gray_matter'][0],
                            results['white_matter'][0],
                            results['fc'][0],
                            results['rs'][0],
                            results['EMOTION'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_third_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
            
        #######################################################################
        # Gray Matter, White Matter, Resting State, and Working Memory
        #######################################################################
        
        if feature == 'gray_matter_white_matter_fc_rs_WM' and (pd.notnull(results['gray_matter'][0])
            and pd.notnull(results['white_matter'][0])
            and pd.notnull(results['fc'][0])
            and pd.notnull(results['rs'][0])
            and pd.notnull(results['WM'][0])):
            
            print("Loading third-level pipeline for %s" % (feature))
            
            data=np.asarray([results['gray_matter'][0],
                            results['white_matter'][0],
                            results['fc'][0],
                            results['rs'][0],
                            results['WM'][0]])
            
            pipeline = joblib.load('./models/' + measure + '_Unadj_' + feature + '_third_level_pipeline_and_results.sav')
            regressor=pipeline['regressor'][0]
            
            if intercept != 'yes':
                print('No intercept: Setting intercept to 0')
                regressor.intercept_ = np.array([0], dtype='float32')
                
            y_pred = regressor.predict(data.reshape(1,-1))
    
        if np.isnan(y_pred):
            results[feature] = y_pred
        else:
            results[feature] = y_pred[0] 
    
    '''        
    #######################################################################
    #######################################################################
    #######################################################################
    # Save results
    #######################################################################
    #######################################################################
    #######################################################################
    '''
    
    results.to_csv(output, header=True, index=False, na_rep='nan')
    
    return results
