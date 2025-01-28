# British Heart Foundation Classification Challenge   
  
Kaggle link: https://www.kaggle.com/competitions/bhf-data-science-centre-ecg-challenge  

The challenge is multi-label classification. We have images of ECGs and we have to classify each with labels. Each ECG may have none, 1 or more than 1 of the labels. The images themselves are photos of printed out ECGs (all synthetic data). The photos are not all good quality.   

## Notes
- Some `.png` files are broken and can't be read: these are listed in `./broken_images`, need to make sure these are not included in training
- Prototyping done here, ultimately being run on Kaggle

## Current Observations

- Resizing the pictures seems to lead to aliasing and pictures become unreadable
- We seem to be predicting every image has 0 for every category: seems maybe a bit unlikely, even if it were to be random, you'd maybe expect some 1s if totally randomly initialised originally? 
- Finding that the raw prediction values for each of the categories is extremely low 
- Validation loss is not decreasing over time: the training of the model accomplishes nothing: possily because of the aliasing effects

## Potential Approaches

| Idea     | Motivation | Issues | Have we tried this? |
|----------|------------|--------|---------------------|
| Find a set of transformations that makes the images better | Improve the quality of the images, possibly extract just the paper region off the page, resize without artifacts | Images are very different; cannot seem to find a consistent set of transformations that improves them | Tried a few things, hasn't gone too well |
| Using a set of pre-convolutions to deal with the image size aliasing problem | Would allow us to downsize a bit more intelligently without losing too much information, have a set of Conv/Pool layers to reduce image size down from something large to 224x224 and then feed to ResNet | Computational expense | Not yet |
| Using VisionTransformers or DoNuT? | Existing implementations of document recognition are out there, maybe one of those will help | Bit of a shot in the dark | Yes, DoNuT was difficult to install so far, but we didn't try that hard |

