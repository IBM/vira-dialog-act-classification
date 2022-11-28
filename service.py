#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import logging.config
import os

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from consts import HF_MODEL_ID, CLASSES_PATH

logging.config.fileConfig('logging.conf')
log = logging.getLogger('services.dialog')

log.info('Initiating service')
app = FastAPI(openapi_url=None)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
df = pd.read_csv(CLASSES_PATH)
model_trained_classes = df.iloc[:, 0].apply(str.strip).to_list()

log.info('Service is ready')


def get_model_predictions(candidates):
    results = pipeline(candidates)
    class_scores = np.array([[label['score'] for label in result] for result in results])
    class_ids = np.argmax(class_scores, axis=1)
    class_scores = np.max(class_scores, axis=1)
    model_classes = [model_trained_classes[class_id] for class_id in class_ids]
    return model_classes, class_scores.tolist()


@app.get("/")
def read_root():
    return "VIRA Dialog-Act Classification"


@app.get("/health")
def read_root():
    return "OK"


@app.get("/classify")
def read_root(text: str):
    return get_model_predictions([text])


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8040)
