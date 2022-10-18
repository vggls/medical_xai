# explainability_with_pytorch

The purpose of this repo is to gather material which is related to explainability algorithms for image data.

(below files are under construction)

We note that at the beginning of each .py file, in the comments section, we have included the sources (theory, code etc) used along with remarks and the main ideas where necessary.

- *heatmap_LIME.py, heatmap_HiResCAM.py, heatmap_DeepLIFT.py* <br/>

    For each XAI algorithm, we compute three main outputs :
    
    - A pixel-level heatmap, named 'attributions', which comes from a direct application of the XAI alg to the image pixels. It is 2-dim (no channels) and has the same size as the image it comes from. Pixel attributions can take both negative and positive values as per xai algorithms construction.
    - A region-level heatmap, named 'heatmap', which emerges by applying ReLU and AvgPooling transformations on 'attributions'. The values range in [0, 1].
    - A list of the 'heatmap' regions in descending order of importance. The list is named 'regions'.

    We note that 'heatmap' and 'regions' will serve as the main tools to develop evaluation metrics for the XAI algorithms. (see morf.py and haas.py below)

- *overlay.py* : In this file we generate a super-imposed version of the original image by adding a weighted version of the XAI algorithm heatmap.

- *example(s).ipynb* : (to be completed) (to use dataloaders)

- *morf.py* : Class which implements the MoRF tenchnnique for heatmap evaluation and calculates the aopc_score.

- *haas.py* : (to be completed)
