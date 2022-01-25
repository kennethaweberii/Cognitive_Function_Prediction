import os
import pandas as pd
import numpy as np

def convert_freesurfer(csv_filename=None,
         aseg_stats_filename='aseg.stats',
         lh_aparc_stats_filename='lh.aparc.stats',
         rh_aparc_stats_filename='rh.aparc.stats'):
    
    if os.path.exists(aseg_stats_filename) and os.path.exists(lh_aparc_stats_filename) and os.path.exists(rh_aparc_stats_filename):
        print("Loading Freesurfer data for %s" % (csv_filename))
        
        #Load aseg
        
        aseg_columns = pd.read_csv(aseg_stats_filename, delim_whitespace=True, skiprows=79, nrows=0)
        aseg_columns = aseg_columns.drop(columns=['#','ColHeaders'])
            
        aseg_data = pd.read_csv(aseg_stats_filename, delim_whitespace=True, header=None, skiprows=80, nrows=45, names=aseg_columns.columns)
        
        aseg_volume = pd.pivot_table(aseg_data, values='Volume_mm3', columns=['StructName'])
        aseg_volume = aseg_volume.reset_index(drop=True)
        
        #Load lh_aparc
        
        lh_aparc_columns = pd.read_csv(lh_aparc_stats_filename, delim_whitespace=True, skiprows=59, nrows=0)
        lh_aparc_columns = lh_aparc_columns.drop(columns=['#','ColHeaders'])
            
        lh_aparc_data = pd.read_csv(lh_aparc_stats_filename, delim_whitespace=True, header=None, skiprows=60, nrows=34, names=lh_aparc_columns.columns)
        
        lh_aparc_data['Thickness_Labels'] = 'lh_' + lh_aparc_data['StructName'] + '_thickness'
        lh_thickness = pd.pivot_table(lh_aparc_data, values='ThickAvg', columns=['Thickness_Labels'])
        lh_thickness = lh_thickness.reset_index(drop=True)
    
        lh_aparc_data['Area_Labels'] = 'lh_' + lh_aparc_data['StructName'] + '_area'    
        lh_area = pd.pivot_table(lh_aparc_data, values='SurfArea', columns=['Area_Labels'])
        lh_area = lh_area.reset_index(drop=True)
        
        #Load rh_aparc
        rh_aparc_columns = pd.read_csv(rh_aparc_stats_filename, delim_whitespace=True, skiprows=59, nrows=0)
        rh_aparc_columns = rh_aparc_columns.drop(columns=['#','ColHeaders'])
            
        rh_aparc_data = pd.read_csv(rh_aparc_stats_filename, delim_whitespace=True, header=None, skiprows=60, nrows=34, names=rh_aparc_columns.columns)
        
        rh_aparc_data['Thickness_Labels'] = 'rh_' + rh_aparc_data['StructName'] + '_thickness'
        rh_thickness = pd.pivot_table(rh_aparc_data, values='ThickAvg', columns=['Thickness_Labels'])
        rh_thickness = rh_thickness.reset_index(drop=True)
    
        rh_aparc_data['Area_Labels'] = 'rh_' + rh_aparc_data['StructName'] + '_area'    
        rh_area = pd.pivot_table(rh_aparc_data, values='SurfArea', columns=['Area_Labels'])
        rh_area = rh_area.reset_index(drop=True)
        
        data = pd.concat([aseg_volume, lh_thickness, lh_area, rh_thickness, rh_area], join='outer', axis=1)    
        
        data = data[['Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Left-VentralDC', 'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'lh_bankssts_thickness', 'lh_caudalanteriorcingulate_thickness', 'lh_caudalmiddlefrontal_thickness', 'lh_cuneus_thickness', 'lh_entorhinal_thickness', 'lh_fusiform_thickness', 'lh_inferiorparietal_thickness', 'lh_inferiortemporal_thickness', 'lh_isthmuscingulate_thickness', 'lh_lateraloccipital_thickness', 'lh_lateralorbitofrontal_thickness', 'lh_lingual_thickness', 'lh_medialorbitofrontal_thickness', 'lh_middletemporal_thickness', 'lh_parahippocampal_thickness', 'lh_paracentral_thickness', 'lh_parsopercularis_thickness', 'lh_parsorbitalis_thickness', 'lh_parstriangularis_thickness', 'lh_pericalcarine_thickness', 'lh_postcentral_thickness', 'lh_posteriorcingulate_thickness', 'lh_precentral_thickness', 'lh_precuneus_thickness', 'lh_rostralanteriorcingulate_thickness', 'lh_rostralmiddlefrontal_thickness', 'lh_superiorfrontal_thickness', 'lh_superiorparietal_thickness', 'lh_superiortemporal_thickness', 'lh_supramarginal_thickness', 'lh_frontalpole_thickness', 'lh_temporalpole_thickness', 'lh_transversetemporal_thickness', 'lh_insula_thickness', 'rh_bankssts_thickness', 'rh_caudalanteriorcingulate_thickness', 'rh_caudalmiddlefrontal_thickness', 'rh_cuneus_thickness', 'rh_entorhinal_thickness', 'rh_fusiform_thickness', 'rh_inferiorparietal_thickness', 'rh_inferiortemporal_thickness', 'rh_isthmuscingulate_thickness', 'rh_lateraloccipital_thickness', 'rh_lateralorbitofrontal_thickness', 'rh_lingual_thickness', 'rh_medialorbitofrontal_thickness', 'rh_middletemporal_thickness', 'rh_parahippocampal_thickness', 'rh_paracentral_thickness', 'rh_parsopercularis_thickness', 'rh_parsorbitalis_thickness', 'rh_parstriangularis_thickness', 'rh_pericalcarine_thickness', 'rh_postcentral_thickness', 'rh_posteriorcingulate_thickness', 'rh_precentral_thickness', 'rh_precuneus_thickness', 'rh_rostralanteriorcingulate_thickness', 'rh_rostralmiddlefrontal_thickness', 'rh_superiorfrontal_thickness', 'rh_superiorparietal_thickness', 'rh_superiortemporal_thickness', 'rh_supramarginal_thickness', 'rh_frontalpole_thickness', 'rh_temporalpole_thickness', 'rh_transversetemporal_thickness', 'rh_insula_thickness', 'lh_bankssts_area', 'lh_caudalanteriorcingulate_area', 'lh_caudalmiddlefrontal_area', 'lh_cuneus_area', 'lh_entorhinal_area', 'lh_fusiform_area', 'lh_inferiorparietal_area', 'lh_inferiortemporal_area', 'lh_isthmuscingulate_area', 'lh_lateraloccipital_area', 'lh_lateralorbitofrontal_area', 'lh_lingual_area', 'lh_medialorbitofrontal_area', 'lh_middletemporal_area', 'lh_parahippocampal_area', 'lh_paracentral_area', 'lh_parsopercularis_area', 'lh_parsorbitalis_area', 'lh_parstriangularis_area', 'lh_pericalcarine_area', 'lh_postcentral_area', 'lh_posteriorcingulate_area', 'lh_precentral_area', 'lh_precuneus_area', 'lh_rostralanteriorcingulate_area', 'lh_rostralmiddlefrontal_area', 'lh_superiorfrontal_area', 'lh_superiorparietal_area', 'lh_superiortemporal_area', 'lh_supramarginal_area', 'lh_frontalpole_area', 'lh_temporalpole_area', 'lh_transversetemporal_area', 'lh_insula_area', 'rh_bankssts_area', 'rh_caudalanteriorcingulate_area', 'rh_caudalmiddlefrontal_area', 'rh_cuneus_area', 'rh_entorhinal_area', 'rh_fusiform_area', 'rh_inferiorparietal_area', 'rh_inferiortemporal_area', 'rh_isthmuscingulate_area', 'rh_lateraloccipital_area', 'rh_lateralorbitofrontal_area', 'rh_lingual_area', 'rh_medialorbitofrontal_area', 'rh_middletemporal_area', 'rh_parahippocampal_area', 'rh_paracentral_area', 'rh_parsopercularis_area', 'rh_parsorbitalis_area', 'rh_parstriangularis_area', 'rh_pericalcarine_area', 'rh_postcentral_area', 'rh_posteriorcingulate_area', 'rh_precentral_area', 'rh_precuneus_area', 'rh_rostralanteriorcingulate_area', 'rh_rostralmiddlefrontal_area', 'rh_superiorfrontal_area', 'rh_superiorparietal_area', 'rh_superiortemporal_area', 'rh_supramarginal_area', 'rh_frontalpole_area', 'rh_temporalpole_area', 'rh_transversetemporal_area', 'rh_insula_area']]
        
        if csv_filename != None:
            print("Saving data as %s" % (csv_filename))
            data.to_csv(csv_filename, header=True, index=False, na_rep='nan')
        
    else:
        print('Error converting Freesurfer')
        data=np.nan
        
    return data
