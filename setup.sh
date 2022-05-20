# Check to see that tensorflow will work on this machine
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# clone the repo
echo "$VAR_pub" > "/root/.ssh/id_ed25519.pub"
echo "$VAR_private" > "/root/.ssh/id_ed25519"

cd /home
chmod 600 /root/.ssh/id_ed25519.pub
chmod 600 /root/.ssh/id_ed25519
git clone git@gitlab.com:Octave2/ml.git /home/ml/
git config --global user.email "tom.moran4@gmail.com"
git config --global user.name "Tom Moran"

envsubst  </root/.gcp/gcp.json >> /root/.gcp/nth-fiber-303315-ee5378f17565.json
envsubst  </root/.kaggle/kaggle_template.json >> /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
gcloud auth activate-service-account dl-sa-495@nth-fiber-303315.iam.gserviceaccount.com --key-file=/root/.gcp/nth-fiber-303315-ee5378f17565.json --project=nth-fiber-303315
mkdir -p datasets/retinopathy/checkpoints/
gsutil -m cp -r "gs://ml-start-$1/train_images_processed/" datasets/retinopathy/
# gsutil -m cp -r "gs://ml-start-$1/test_images_processed/" datasets/retinopathy/
gsutil cp "gs://ml-start-$1/trainLabels.csv" datasets/retinopathy/
