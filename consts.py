#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import os

DATASET_DIR = 'dialog_act_dataset'
MODEL_DIR = 'dialog_act_model'
CLASSES_FILE = 'dialog_acts.csv'
CLASSES_PATH = os.path.join(DATASET_DIR, CLASSES_FILE)

BASE_MODEL = 'roberta-large'
N_EPOCHS = 15
# BASE_MODEL = 'roberta-base'
# N_EPOCHS = 1
LEARNING_RATE = 5e-6

HF_DATASET_ID = "ibm/vira-dialog-acts-live"
HF_MODEL_ID = f'ibm/{BASE_MODEL}-vira-dialog-acts-live'
