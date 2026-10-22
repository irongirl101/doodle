import numpy as np 
import os 
import utils
from sklearn.model_selection import train_test_split

BITMAP = 800
RANDOM_SEED = 42 

class messagedata: 
    def binarize(self,x): # this sets all the pixel values above 0 to be 225 -- so its black or white.
        x[x>0] = 225 
        return x 

    def get_data(self): 
        return(self.X,self.y)

    def get_train(self): 
        return(self.X_train,self.y_train)

    def get_test(self): 
        return(self.X_test.self.y_test)

    def get_dev(self): 
        return(self.X_dev, self.y_dev)

    def __init__(self, folder = 'dataset', binarize = False):
        self.folder = folder
        self.X = np.zeros((0,BITMAP))
        self.y = []

        self.X_train = np.zeros((0,BITMAP))
        self.y_train = [] 

        self.X_test = np.zeros((0,BITMAP))
        self.y_test = [] 

        self.X_dev = np.zeros((0,BITMAP))
        self.y_dev = [] 

        max_examples = 20000 # max examples from each class -- limiting to avoid OOMs. 
        for file in enumerate(os.listdir(self.folder)): 
            path = self.folder + file
            name = file[:-4]

            x = np.load(path)
            x = x[:max_examples,:]
            print('Shape', x.shape)

            if binarize: 
                x = self.binarize(x)

            y = [name] * x.shape[0]

            X_train,X_eval,y_train, y_eval = train_test_split(x, y, test_size=0.2,random_state=RANDOM_SEED)
            X_dev, X_test, y_dev, y_test = train_test_split(X_eval,y_eval, test_size=0.2,random_state=RANDOM_SEED)

            self.X_train = np.concatenate((self.X_train,X_train), axis = 0)
            self.y_train += y_train

            self.X_dev = np.concatenate((self.X_dev,X_dev), axis = 0)
            self.y_dev += y_dev 

            self.X_test = np.concatenate((self.X_test,X_test), axis = 0)
            self.y_test += y_test

        self.Y = np.array(self.Y)
        self.y_train = np.array(self.y_train)
        self.y_dev = np.array(self.y_dev)
        self.y_test = np.array(self.y_test)

        print('Full', self.X.shape)
        print('Train',self.X_train.shape)
        print('Test',self.X_test.shape)
        print('Dev',self.X_dev.shape)

def main():
    data = messagedata()

if __name__ == '__main__': 
    main()



        

        

    
