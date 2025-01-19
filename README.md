# British Heart Foundation Classification Challenge   
  
Kaggle link: https://www.kaggle.com/competitions/bhf-data-science-centre-ecg-challenge  

The challenge is multi-label classification. We have images of ECGs and we have to classify each with labels. Each ECG may have none, 1 or more than 1 of the labels. The images themselves are photos of printed out ECGs (all synthetic data). The photos are not all good quality.   

## Notes
- Some `.png` files are broken and can't be read: these are listed in `./broken_images`, need to make sure these are not included in training
- Prototyping done here, ultimately being run on Kaggle

# Current Thoughts

- Resizing the pictures seems to lead to aliasing and pictures become unreadable: this maybe needs a look at it 
- We seem to be predicting every image has 0 for every category: seems maybe a bit unlikely, even if it were to be random, you'd maybe expect some 1s if totally randomly initialised originally? 
- Finding that the raw prediction values for each of the categories is extremely low 
- Validation loss is not decreasing over time: the model just doesn't really seem to work?