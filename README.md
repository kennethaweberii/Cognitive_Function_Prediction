# Cognitive Function Prediction

<p align="center">
<img src="https://github.com/kennethaweberii/Cognitive_Function_Prediction/blob/main/cognitive_function_prediction.jpg" width="750">
</p>

#### This model was developed using Nilearn (https://nilearn.github.io) and Sklearn (https://scikit-learn.org/)

### Installation

1. Install conda: https://www.anaconda.com/
2. Download Cognitive_Function_Prediction code and unzip if necessary. 
3. Download the models folder here: https://office365stanford-my.sharepoint.com/:f:/g/personal/kenweber_stanford_edu/Ekhu_md4SBFEhov3ssODvyQB44wmznIyGcN1Gk6l8U4Skg?e=AB0u8s
    * File is >100 MB and too big for GitHub
4. Untar models.tar.gz and move the models folder into ./Cognitive_Function_Prediction/
5. In the command line, navigate to the Cognitive_Function_Prediction folder
6. Create a conda environment:
    * conda env create -f environment.yml
7. Activate the conda environment:
    * conda activate Cognitive_Function_Prediction
8. Run predict_cognitive_function.py specifying the path to the files for the respective features

### Image Formatting Requirements

1. For the gray matter featuers, the Freesurfer gray matter features (aseg.stats, lh.aparc.stats, and rh.aparc.stats) need to be in single csv file. Use convert_freesurfer_stats_to_csv.py to convert these files to the correct format.
2. For the white matter features, the fractional anisotropy 3D image should be in FMRIB58 1 mm<sup>3</sup> space.
3. For the resting state functional connectivity features, the preprocessed resting state timeseries 4D images should be in MNI152 2 mm<sup>3</sup> space.
4. For the resting state frequency domain and graph measures features, the 3D images should be in MNI152 2 mm<sup>3</sup> space.
	* These images were created using functions available in AFNI (https://afni.nimh.nih.gov/)
5. For the task-evoked features (Working Memory, Category-Specific Representations, Gambling, Language Processing, Social Cognition, Relational Processing, and Emotion Processing), the images (3D) should be in MNI152 2 mm<sup>3</sup> space.

#### FMRIB58 1 mm<sup>3</sup> space and MNI152 2 mm<sup>3</sup> space template images availabled in FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

### Citing Cognitive Function Prediction Model

If you use this model in your work, please cite:

* K. A. Weber II, T.D. Wager, A. Rosen, Y. K. Ashar, C. S. W. Law, G. Gilam, P. A. Upadhyayula, S. Zhu, S. Banerjee, G. H. Glover, T. J. Hastie, and S. Mackey. Predicting Cognitive Function with Multimodal Brain MRI. (In Review).

