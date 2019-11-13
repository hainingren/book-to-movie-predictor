from argparse import ArgumentParser
import logging
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy import interp
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import sys
import unicodedata

import fasttext

logging.basicConfig(level="INFO")

class TextPreprocessor():
    """ General class for preprocessing text.Specify options within
    """
    def __init__(self, lower=True):
        self.lower = lower
        
    def strip_html(self,text):
        return re.sub(r"<[^>]*>", " ", text)
            
    def preprocess_text(self,text):
        text = self.strip_html(text)
        if self.lower==True:
            text = text.lower()
        return text
    
class BookMovieClassifier():
    """FastText Binary Classifier of Prob(Book) gets turned into a movie
    """
    def __init__(self, textPreprocessor, trainPath=None, testPath=None):
        self.classes = ['not_movie','movie']
        self.numClasses = len(self.classes)
        self.fastTextLabels = ['__label__{0}'.format(str(c)) for c in self.classes]
        
        self.trainPath = trainPath
        self.testPath = testPath
        self.model = None
        self.textPreprocessor = textPreprocessor
        
    def load_dataset(self,t='train'):
        assert(t in set(['train','test']))
        
        if t=='train':
            df = pd.read_csv(self.trainPath,encoding='utf-8')
        elif t=='test':
            df = pd.read_csv(self.testPath, encoding='utf-8')
        else:
            assert(t in set(['train','test']))
            
        df['description'] = df.description.apply(lambda x: self.textPreprocessor.preprocess_text(x))
        return df
    
    def save_fasttext_dataset(self,df,t):
        assert(t in set(['train','test']))
        with io.open('data/derived/book_movie_labels_{0}.fasttext'.format(t), 'w', encoding='utf-8') as dataset:
            for _,row in df.iterrows():
                y = row.is_movie
                y = [y]
                sample = row.description
                targets = ' '.join('%s%s' % ('__label__', str(label)) for label in y)
                #dataset.write(targets + ' ' + sample + '\n')
                dataset.write(unicodedata.normalize('NFKD', targets + ' ' + sample + '\n'))#.encode('ascii', 'ignore')))

    def prepare_datasets(self):
        logging.info("Loading and converting datasets to fastText format")
        self.save_fasttext_dataset(self.load_dataset('train'),'train')
        self.save_fasttext_dataset(self.load_dataset('test'),'test')
        
    def train(self,opts=None):
        self.model = fasttext.train_supervised('data/derived/book_movie_labels_train.fasttext',**opts)

    def load_model(self,modelPath):
        logging.info("Loading model")
        self.model = fasttext.load_model(modelPath)
    
    def predict(self, text, returnBinary=True):
        """
        Return a list of predictions from a list of text:
        # Arguments
            - text: String or list of Strings
            - returnBinary: if True, returns positive class predictions, e.g. [0.4,.5,..]
                            if False, returns [{'__label__0': 0.6, '__label__1': 0.4},...]
        """
        if type(text)==str:
            text = [text]
        
        text = list(map(lambda x: x.replace('\n',' '), text))
            
        labelList,probList = self.model.predict(text,
                                           k=self.numClasses)
        
        labelDict = [{c: p  for c,p  in zip(labels, probs)} for labels,probs in zip(labelList, probList)]
        
        if not returnBinary:
            return labelDict
        
        return [p['__label__1'] for p in labelDict]
    
    def calculate_metrics(self, yScore,yPred, precisionSelect=0.3):
        """ Calculate accuarcy metrics.
        # Arguments:
            - yScore: list of true values of each book
            - yPred: predicted list of values
            - precisionSelect: return the threshold at which model precision is precisionSelect
        """

        fpr, tpr, thresh = roc_curve(yScore, yPred)
        roc_auc = auc(fpr, tpr)

        precision, recall, thresh = precision_recall_curve(yScore, yPred)
        avg_pre = average_precision_score(yScore, yPred)

        # with a precision  of 0.3, we would correctly make a movie into a best-sellar 
        threshSelect = thresh[sum(precision<=precisionSelect)]

        return {'roc': roc_auc,
                'avg_pre': avg_pre,
                'thresh_select': threshSelect}

    def evaluate(self):
        dfTrain = pd.read_csv(self.trainPath,encoding='utf-8')
        dfTest = pd.read_csv(self.testPath,encoding='utf-8')

        return self.calculate_metrics(dfTrain['is_movie'],self.predict(list(dfTrain['description']))),\
               self.calculate_metrics(dfTest['is_movie'],self.predict(list(dfTest['description'])))

def train(args):
    parser = ArgumentParser()
    parser.add_argument("-config", type=str, required=True,
                        help="Relative path to model config")
    parser.add_argument("-model-name", type=str, required=True,
                        help="Name of file to dump model to in weights/ folder")
    args = parser.parse_args()

    # Model trainer or evaluator
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = json.load(f)
    
    textPreprocessor = TextPreprocessor()
    model = BookMovieClassifier(textPreprocessor,
                                trainPath=config['trainPath'],
                                testPath=config['testPath'])
    #Train the model
    logging.info("Training Model")
    model.prepare_datasets()
    model.train(opts=config['fastTextOpts'])
    model.model.save_model('weights/{0}'.format(args.model_name))

    #Evaluate the model
    logging.info("Evaluating Model")
    trainEval, testEval = model.evaluate()
    evalDict = {'train': trainEval, 'test': testEval}
    print(evalDict)
    with open('weights/{0}.results'.format(args.model_name),'w') as f:
        json.dump(evalDict,f)

"""
#### USAGE EXAMPLE ###

python fasttext_model.py -config config/train_config.json -model-name fasttext_best

"""
if __name__ == '__main__':
    train(sys.argv[1:])
    



    