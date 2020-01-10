import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
work_dir = os.path.abspath(".")

class KerasTest:
    def __init__(self):
        self.model = load_model(os.path.join(work_dir, "src\\test\\resources\\inference\\keras\\model_ndarray_in_ndarray_out.h5"))

    def test(self,inputarray):
        arr = self.model.predict(inputarray)
        return arr
        print("my_test---------------->"my_test)
 my_test = np.array(([100, 55, 555, 1000], [1, 0, 5, 10]))
print("my_test%%%%%%%%%%%%%%%%%%%%%%%"my_test)
objKeras = KerasTest ()
arr = objKeras.test(my_test)
print(arr)