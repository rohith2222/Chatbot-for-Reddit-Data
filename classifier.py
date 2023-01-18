import pickle
import random
import warnings


warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from smaberta import TransformerModel
import torch
random.seed(1)
np.random.seed(1)
#pd.set_option('precision', 0)
torch.cuda.manual_seed(1)
torch.manual_seed(1)

class Classifier:
    model = TransformerModel('roberta', 'roberta-base', num_labels=6,
                             location="./saved_model/")
    topics = {"politics": 0, "environment": 1, "technology": 2, "healthcare": 3, "education": 4, "chitchat": 5}

    def classify(self, text):
        preds, model_outputs = (self.model.predict([text]))

        for k, v in self.topics.items():
            if (v in preds):
                   return k

    def train(self):

        train_df = pd.read_csv("processed_data.csv")
        test_df = pd.read_csv("processed_data.csv")
        train_df.head()

        lr = 1e-5
        epochs = 5

        model = TransformerModel('roberta', 'roberta-base', num_labels=6, reprocess_input_data=True,
                                 num_train_epochs=epochs, learning_rate=lr, output_dir='./new_model/',
                                 overwrite_output_dir=True, fp16=False)

        model.train(train_df['text'], train_df['label'])

        result, model_outputs, wrong_predictions = model.evaluate(test_df['text'], test_df['label'])
        preds = np.argmax(model_outputs, axis=1)
        correct = 0
        labels = test_df['label'].tolist()
        for i in range(len(labels)):
            if preds[i] == labels[i]:
                correct += 1
        accuracy = correct / len(labels)
        print("Accuracy: ", accuracy)







if __name__ == '__main__':
    c = Classifier()
    text = input()
    print(c.classify(text))


