# mtad-tf
Implementation of MTAD-TF: Multivariate Time Series Anomaly Detection Using the Combination of Temporal Pattern and Feature Pattern

https://www.hindawi.com/journals/complexity/2020/8846608/

Other papers

Omnianomaly:

https://www.researchgate.net/publication/334717291_Robust_Anomaly_Detection_for_Multivariate_Time_Series_through_Stochastic_Recurrent_Neural_Network

Notes:

-train and test data is generated to GCP

-models are stored to GCP

-"anomaly_detection" is my GCP storage

-it takes 4 hours to train 1 model on 16vCPU 16GB(not this much memory is needed)

-I used some very minor utility content from BERT. This is why I put BERT Licence

-Do not use dropout. I found it makes loss higher

-I am discussing SMD only, but MSL and SMAP should be used as well

Training steps:

1. Download ServerMachineDataset from https://github.com/NetManAIOps/OmniAnomaly
Put ServerMachineDataset here at the root of the project

2. Prepare tfrecords train and test data

Train:

python prepare_data.py --files_path=ServerMachineDataset/train --tfrecords_file=gs://anomaly_detection/mtad_tf/data/train/{}.tfrecords

Test:

python prepare_data.py --files_path=ServerMachineDataset/test --label_path=ServerMachineDataset/test_label --tfrecords_file=gs://anomaly_detection/mtad_tf/data/test/{}.tfrecords

3. Train model for each machine dataset, 28 of them. Paper says 100 epochs, so I use 78k steps

python training.py --action=TRAIN --train_file=gs://anomaly_detection/mtad_tf/data/train/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1 --num_train_steps=78000 --learning_rate=0.001

4. Produce RMS loss file - RMS_loss.csv. Use this file to see losses and calculate threshold using POT

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_tf/data/test/machine-1-1.tfrecords --prediction_task=RMS_loss --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1

5. Use my another repository EVT_POT to calculate threshold. RMS_loss.csv is an input for that

6. Use notebook to see RMS_loss.csv, initial threshold and anomaly threshold. Adjust those values in the notebook 


Anomaly Evaluation

I do not use Estimator EVAL to do this since I use an adjustment procedure. I use this same simple omnianomaly approach. If we predicted anomaly and it is within some anomaly segment, whole segment becomes correctly predicted. I do not use a prediction latency (delta) value as in some other papers. Also, I think practically, if we flagged anomaly earlier compare to actual anomaly label, it could be considered hit maybe using some delta as well. I did not do this way here. Please see samples folder for evaluations for machine dataset 1-1, 1-2 and 1-3

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_tf/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1 --threshold=1.5547085 --prediction_task=EVALUATE


Anomaly Prediction

This command creates Anomaly.csv in the local folder. You must provide threshold value

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_tf/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1 --threshold=1.5547085
