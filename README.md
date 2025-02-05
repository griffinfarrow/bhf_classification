# British Heart Foundation Classification Challenge   
  
Kaggle link: https://www.kaggle.com/competitions/bhf-data-science-centre-ecg-challenge  

The challenge is multi-label classification. We have images of ECGs and we have to classify each with labels. Each ECG may have none, 1 or more than 1 of the labels. The images themselves are photos of printed out ECGs (all synthetic data). The photos are not all good quality.   

## Notes
- Some `.png` files are broken and can't be read: these are listed in `./broken_images`, need to make sure these are not included in training
- Prototyping done here, ultimately being run on Kaggle

## Models 

### 1) Object Detection Model 
This is a model based on Fast-R-CNN that has been pre-trained on images that we annotated ourselves. It pretty effectively identifies a document in an image and draws a bounding box around it. This can restrict focus on the document in the image. The model training and testing are found in `object_detection_model/` and the raw model weights are found in `model/train_ecg_object_detection_model.pth`. 

This model does work well, but we have sort of moved away from it in favour of segmentation models that more specifically isolate the document in the image.

### 2) Segmentation Model 

This is an attempt to build segmentation models that find the ECG in the image and then focuses in on it. This should mean that we lose a lot of the irrelevant background.  

My custom attempt at doing this was converting the "Document Object Detection" model built using Fast-R-CNN into a mask prediction model. We annotated some data using `Label-Studio` and tried to train the model on it using `train_seg_model_untransformed.ipynb`.  

This seemed to not work at all. The mask loss predicted by the model was always zero and the masks were nonsensical. I'm still not entirely sure whether this was to do with the annotated data that we gave the model being terrible, or just simply not enough, an error in model definition or an error in model training.  

I've found an alternative segmentation model that seems to work really well. This in in `external_seg_model.ipynb`. This comes from https://github.com/veb-101/Document-Segmentation-using-Pytorch-DeepLabV3/tree/main and the model weights are on Kaggle at https://www.kaggle.com/models/abdxlhaxk/document-detection. On preliminary testing, this model seems to be very good at the document segmentation, so we're going to use it as part of the full ECG classification pipeline.

## Past Attempts 

| Idea | What?            |Code?| Results                   | 
|------|------------------|-----|---------------------------|
| End-to-end approach | No pre-processing of images. Just use `Albumentations` for transformations to the data, then feed into a pre-trained `ResNet-18` model for fine-tuning. Last layer of `ResNet-18` adapted for multi-label classification instead | `ecg_classification_model/bhf-heart-rhythm-classification.ipynb` | Didn't seem to work. Produced nearly 0 probability for all class labels and training didn't improve things at all. Possibly because when the images are resized from their original $1900 \times 3460$ to $224 \times 224$, it really badly aliases them and you end up not being able to actually see anything in the data at all. |