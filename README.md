## *Understanding the Role of Weather Data for Earth Surface Forecasting using a ConvLSTM-based Model*

This repository accompanies the paper *Understanding the Role of Weather Data for Earth Surface Forecasting using a ConvLSTM-based Model*. 

![](supl/overview_conv_lstm.gif "ConvLSTM Training Procedure")

## Structure
1. `code`: it will include the code used to train the model
2. `models`: it will include the weights of the trained model used to obtain the predictions reported in the paper and in this repository
3. `supl`: supplementary materials which are not be included in the paper, read `supl\readme.md` for a full description
   1. `supl/predictions`: RGB & NIR predictions for five random samples together with the corresponding ground truth
   2. `supl/simulations`: RGB & NIR predictions for five random samples, shown as animations, obtained by individually perturbing each of the three meteorological conditions (i.e. rainfall, sea level pressure and temperature)
