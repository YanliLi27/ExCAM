# ExCAM

## The code for "Towards informative saliency maps for medical images" 

> Thanks to https://github.com/frgfm/torch-cam and https://github.com/lucidrains/vit-pytorch for their functions.

> Currently, support for MNIST, ILSVRC2012, Cats&Dogs. ESMIRA (private data) is not supported as it includes the information of patients.


## Use:

1. main.py provides the function of analyzer, predictor and evaluator.


2. In main.py, examples were given for generating CAMs of MNIST, ILSVRC2012, Cats&Dogs, with the default paths.


3. **Change paths and models, please see the generators/main_generator.py **

    > (1) for path use: path_generator.py. 
    
    > (2) for model use: model_generator.py (or any other models you have). 
    
    > (3) for dataset use: dataset_generator.py. 
    any combination is allowed, if you understand the type of data.
 
 
 4. **Add more CAM methods, please see the /cam_components/\*cam.py and  /cam_components/\*cam_pred.py**
 
    > (1) \*cam.py is for the analyzer, \*cam_pred.py is for the CAM generation.
    
    > (2) If you'd like to change the core part, see /cam_components/core/base_cam_\*.py 
    
    > (3) Highly recommend to avoid using score CAM and the variants, as they takes too much time. 

5. **Change the functions for importance matrices and evaluation, see /main_func/*.py**


6. For the output, you can create a dir named output for collection.



## Examples:

> IM scaling, when your CAM targeting dogs
![Example](/1.png "IM scaling, when your CAM targeting dogs")




