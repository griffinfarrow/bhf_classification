# British Heart Foundation Classification Challenge   
  
Kaggle link: https://www.kaggle.com/competitions/bhf-data-science-centre-ecg-challenge  

The challenge is multi-label classification. We have images of ECGs and we have to classify each with labels. Each ECG may have none, 1 or more than 1 of the labels. The images themselves are photos of printed out ECGs (all synthetic data). The photos are not all good quality.   

## Notes
- Some `.png` files are broken and can't be read: these are listed in `./broken_images`, need to make sure these are not included in training
- Prototyping done here, ultimately being run on Kaggle

# Approaches Tried

## Using `ResNet18` pre-trained and fine-tuned on our data
- We've managed to get this to run 
- Seems to not perform all that well: we get stuck with a validation accuracy of about 82%
- The way that we're calculating accuracy is adding up the correct predictions from all 5 categories: is it consistently underperforming with one of the categories?
- Alternatively, is the way that we're pre-processing the data (transformations) actually not very sensible?
- Or is ResNet18 just not the way to go?
  
It does seem to be the case that the CLAHE contrast thing we're applying is washing out the images so there is nothing on them and the resizing is introducing artifacts.   
  
I think something might be happening with class imbalance/some classes not having any predictions made for them. With significant changes, I still got the exact same accuracy score (seems unlikely). I'm getting the error that some classes have zero predictions made for them, which is setting a lot of the metrics to zero. This seems really unlikely, so I suspect something a bit fishy is happening. This could do with a bit more investigation.