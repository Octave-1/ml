# https://www.kaggle.com/competitions/diabetic-retinopathy-detection/
using a tensorflow CNN & scikit learn to predict diabetic retinopathy
in patients, given images of both their eyes.
- training was carried out using a VastAI instance on a custom docker
container (see docker file in repo)
- the images were pre-processed & downsized (see pre_process.py)
- the pre-processed images were stored on GCP & downloaded for
model training (see setup.sh)
- this version achieves a quadratic kappa score of ~ 0.7 on the
private leaderboard
