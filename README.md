# British Heart Foundation Classification Challenge   
  
Kaggle link: https://www.kaggle.com/competitions/bhf-data-science-centre-ecg-challenge  

The challenge is multi-label classification. We have images of ECGs and we have to classify each with labels. Each ECG may have none, 1 or more than 1 of the labels. The images themselves are photos of printed out ECGs (all synthetic data). The photos are not all good quality.   

## Notes
- Some `.png` files are broken and can't be read: these are listed in `./broken_images`, need to make sure these are not included in training
- Biggest bottleneck is in loading the initial images. The actual ECG part is a tiny part of the overall image. We preprocess by segmenting and just isolating the ECG, then downsize the image and save that. This reduces the total size of all the images from ~150GB to ~1.5GB. This allows us to store it all in memory and drastically speeds up processing time