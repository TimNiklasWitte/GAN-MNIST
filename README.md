# GAN: Generative Adversarial Networks


## Evaluation

### Loss

<img src="./plots/GeneratorDiscriminatorLoss.png" width="650" height="200">

### Discriminator Loss

<img src="./plots/DiscriminatorLosses.png" width="650" height="200">

### Discriminator Accuracy
<img src="./plots/DiscriminatorAccuracy.png" width="650" height="200">


### Generated images while training
The GAN was trained for 150 epochs:
Each column represents a noise vector which is transformed by the generator into an image.
There are ten different noise vectors.
The i^th row represents the model's state at epoch i * 10.

<img src="./plots/GeneratedImgsWhileTraining.png" width="650" height="1000">

### Generated images
Let's generate 100 images after training the GAN i.e. after 150 epochs.

<img src="./plots/GenerateImgs.png" width="650" height="650">
