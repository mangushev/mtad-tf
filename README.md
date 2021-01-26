# mtad-tf
Implementation of MTAD-TF: Multivariate Time Series Anomaly Detection Using the Combination of Temporal Pattern and Feature Pattern

https://www.hindawi.com/journals/complexity/2020/8846608/

THIS IS A DRAFT. I didn't finish any evaluation

I am discussing SMD only, but MSL and SMAP should be used as well

My setup:
-train and test data is generated to GCP
-models are stored to GCP
-"anomaly_detection" is my GCP storage

Steps:

1. Download ServerMachineDataset from https://github.com/NetManAIOps/OmniAnomaly

2. Prepare tfrecords train and test data
train:
python prepare_data.py --files_path=ServerMachineDataset/train --tfrecords_file=gs://anomaly_detection/mtad_tf/data/train/{}.tfrecords
test:
python prepare_data.py --files_path=ServerMachineDataset/test --label_path=/work/datasets/ServerMachineDataset/test_label --tfrecords_file=gs://anomaly_detection/mtad_tf/data/test/{}.tfrecords

3. Train model for each machine dataset, 28 of them. Paper says 100 epochs, so I use 78k steps

python training.py --action=TRAIN --train_file=gs://anomaly_detection/mtad_tf/data/train/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1 --num_train_steps=78000 --learning_rate=0.001

4. Produce RMS loss file - RMS_loss.csv. Use this file to see losses and calculate threshold using POT
python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_tf/data/test/machine-1-1.tfrecords --prediction_task=RMS_loss --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1

5. Use my another repository EVT_POT to calculate threshold. RMS_loss.csv is an input for that
