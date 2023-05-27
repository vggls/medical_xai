## Short summary

In this use case we consider breast tissue images taken from the [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis) dataset. For a trained ResNet50 variant (no GAP layer) we generate the GradCAM and HiResCAM test set attribution maps and compare their quality in terms of the AOPC and Max Sensitivity metrics.

We observe that
- for AOPC HiResCAM outperforms GradCAM on average wrt heatmap the size
- for Max Sensitivity HiResCAM outperforms GradCAM over all radii tested (and consequently on average as well)

## ResNet50 Testing Results

<p float="left">
     <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/22bf70cb-c32f-4a7e-b38b-926e05be492e" height="145" width="270" />
     <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/720540a5-6a27-4118-ae1e-cdb51582dcf3" height="230" width="320" />
   </p>
   
## Max Sensitivity Results
<p float="left">
     <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/f7ccca5c-f371-40da-9154-4921b9eb1b3e" height="230" width="550" />
   </p>
   
## AOPC Results
<p float="left">
     <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/383e9d9a-dea8-4e3f-9103-15b931df82e6" height="140" width="250" />
   </p>
   
<p float="left">
     <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/1ca339a3-2591-4a2f-b46d-089d574460da" height="230" width="450" />
     <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/d959d478-243a-4d4e-b2b1-a105aae4ceab" height="230" width="450" />
   </p>
   
<p float="left">
   <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/49f78edc-dbbd-48d0-89ae-3a01fce1cda9" height="230" width="450" />
   <img src="https://github.com/vggls/M.Sc._AI_Thesis/assets/55101427/206fb98d-3bdb-4979-abf3-ad65c3d32dfa" height="230" width="450" />
 </p>
