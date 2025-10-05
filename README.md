# Crime-Detection 
Abstract—With an increase in number of surveillance cameras installed in the public, it has become a more difficult mission for the
authorities to constantly monitor the screen. To solve this issue, we present a ”Crime Analysis Model” which is able to classify 10
different types of crime efficiently using the convolutional neural network (CNN) given any surveillance video footage as its input.
We have utilized two separate deep learning models within the Crime Analysis model, namely (1) anomaly detector (2) crime classifier
(which is trained based on the human-action recognition). The Crime Analysis model takes any form of surveillance video footage as its
input. The input first gets passed to the Anomaly Detector which helps to distinguish time frames at which anomalies occur. The
corresponding time frames at which the anomalies occur gets passed to the crime classifier. The change of position of the skeletons
within the video are analyzed and the model outputs one of the following crime categories: (1) Abuse (2) Arson (3) Assault (4) Burglary
(5) Fighting (6) Robbery (7) Shooting (8) Shoplifting (9) Stealing (10) Vandalism

<img width="1686" height="1013" alt="CrimeClassifierOutputSuccess" src="https://github.com/user-attachments/assets/ead9a60d-f670-4658-a10c-f9f0ed34159d" />
