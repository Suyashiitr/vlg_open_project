# vlg_open_project
Project : AI GENERATED IMAGE DETECTION

A train dataset of 5250 examples with 1200 columns is used for training a model to identify whether an image is real or fake. Labels provided are 0 or 1 for each image. The images are 20x20 images with RGB channels ( therefore 3 channels making the dimensions 20x20x3).
The test dataset is 2250 examples of same format (1200 columns)

6 images have been plotted using matplotlib as an example of how they look after clamping the pixel values between 0 and 1.

There are 3850 '0' labels which means the dataset is not balanced and has much higher ratio of one particular label.

The Test and Train datasets have been taken from https://bitgrit.net/competition/18

Validation dataset has been created using sklearn.model_selection_train_test_split with the size being 20% of the train dataset. Accuracy score and f1 score mentioned have been calculated on the validation dataset. 

The first model is Convolutional Neural Network with 2 Conv2d layers , 1 max pooling layer and 3 fully connected layers :
Net(
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (hl1): Linear(in_features=972, out_features=243, bias=True)
  (hl2): Linear(in_features=243, out_features=122, bias=True)
  (hl3): Linear(in_features=122, out_features=61, bias=True)
  (hl4): Linear(in_features=61, out_features=2, bias=True)
)
Batch size used is 64 with 30 epochs and learning rate 0.0001 Adam optimizer , yielded 0.312 loss and 74.1% accuracy on validation dataset. This was the best result after trying a few neural nets ( their results are mentioned in last cell of VLG Project CNN.ipynb file.

The second model is Vision Transformer  with specifications mentioned below :
VisionTransformer(
  (patch_embedding): Conv2d(3, 128, kernel_size=(5, 5), stride=(5, 5))
  (transformer): TransformerEncoder(
    (layers): ModuleList(
      (0-5): 6 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (fc): Linear(in_features=128, out_features=2, bias=True)
)

Patch size is 5x5 , batch size = 64 , learning rate 0.0001 , 25 epochs giving a loss of 0.089 and accuracy score 89.33%
A few results after tuning hyperparameters are again given in last cell of VLG Project(ViT).ipynb file.

Patch size of 10x10 resulted in faster training but the accuracy score on validation set was 84%.

256 embedding dimensions and patch size 10x10 resulted in lowest loss (0.005) and accuracy score was 88% on validation set while f1 score was 71%.

Losses with each epoch are plotted in python notebook files for both models.

loss vs epoch plot of lowest loss model is given below :

<img width="549" alt="Screenshot 2023-06-19 at 2 51 13 AM" src="https://github.com/Suyashiitr/vlg_open_project/assets/114506717/90f8ae8c-6adc-4eec-b585-5aedc756827c">



Observation : Vision Transformer took more time to train for the same number of epochs and same batch size compared to CNN model but gave lesser loss and therefore higher accuracy on validation dataset. The difference between accuracies by both the model is of 15% (best results compared).
Changing the patch size in Vision Transformer had a significant impact. Lower patch size resulted in slower training while increasing the patch size resulted in faster training and lower loss ( however, the accuracy and f1_score on validation dataset wasn't higher than patch_size = 5 case)
ML algorithms using Sci-kit learn were implemented. These were support vector machines( svm) , MLPClassifier, XG Booster (XGB) and K Nearest Neighbors (KNN). Their performance is :


<img width="493" alt="Screenshot 2023-06-21 at 12 19 38 PM" src="https://github.com/Suyashiitr/vlg_open_project/assets/114506717/584f049c-285d-4fd2-bdb2-19b21f5a19f6">


The best model was the K Nearest Neighbors algorithm when it comes to the F1 Score (71) while highest accuracy score on validation dataset was given by VIT model(89%). The highest bitgrit score is 92.7%
