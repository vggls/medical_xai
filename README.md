The purpose of this repo is to gather code that is related to explainability algorithms and evaludation metrics.

We note that at the beginning of each .py file, in the comments section, we have included the sources used (theory, code etc) along with remarks and the main ideas, where necessary.

- **heatmaps.py** <br/>

    For each XAI algorithm, we compute three main outputs :
    
    - A pixel-level heatmap, named 'attributions', which comes from a direct application of the XAI alg to the image pixels. It is 2-dim np.ndarray (no channels) and has the same size as the image it comes from.  For DeepLIFT and LIME the attribution values are in [-1, 1] while for HiResCAM in [0, 1].
    - A region-level heatmap, named 'heatmap', which emerges by applying AvgPooling transformation on 'attributions'.
    - A list of the 'heatmap' regions in descending order of importance. The list is named 'regions'.

    We note that 'heatmap' and 'regions' will serve as the main tools for the calculation of the xai evaluation metric called AOPC, in *morf.py*

- **overlay.py** : In this file we generate a super-imposed version of the original image by adding a weighted version of the XAI algorithm heatmap.

- **morf.py** : For a given tensor and model, the 'MoRF' class implements the MoRF tenchnnique for heatmap evaluation and calculates the AOPC score. The file also includes a method that extends the calculation on a dataset level, when the data are called via Dataloaders.

- **haas.py** : Calculation of the HAAS score for evaluation of XAI algorithms
