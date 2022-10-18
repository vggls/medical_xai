# explainability_with_pytorch

The purpose of this repo is to gather material which is related to explainability algorithms for image data.

(below files are under construction)

We note that at the beginning of each .py file, in the comments section, we have included a list of the sources (theory, code etc) that the file was build on.

- *heatmap_LIME.py, heatmap_HiResCAM.py, heatmap_DeepLIFT.py* <br/>

    For each XAI algorithm, we compute three main outputs :
    
    - A pixel-level heatmap, named 'attributions', which comes from a direct application of the XAI alg to the image pixels. It is 2-dim (no channels) and has the same size as the image it comes from. Pixel attributions can take both negative and positive values.
    - A region-level heatmap, named 'heatmap', which comes from AvgPooling and ReLU transformation on 'attributions'. The values range in [0, 1].
    - A list of the 'heatmap' regions in descending order of importance. The list is named 'regions'.

    We note that 'heatmap' and 'regions' will serve as the main tools to develop evaluation metrics for the XAI algorithms. (see morf.py and haas.py below)

- *overlay.py* : (to be completed)

- *example(s).ipynb* : (to be completed) (to use dataloaders)

- *morf.py* : Class which implements the MoRF tenchnnique for heatmap evaluation and calculates the aopc_score.

- *haas.py* : (to be completed)
