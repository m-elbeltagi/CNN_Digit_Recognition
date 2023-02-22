import pandas as pd
import numpy as np
from PIL import Image as im
from Digit_Recognition_training_model import *



image_test_flatenned = pd.read_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_Digit_Recognizer\test.csv')

## reshape to be in format that model takes
test_images_tensor = image_test_flatenned.to_numpy()
test_images_tensor = np.reshape(test_images_tensor, (len(image_test_flatenned), 28, 28))
test_images_tensor =  torch.from_numpy(test_images_tensor).float()
test_images_tensor = test_images_tensor.view(len(image_test_flatenned),1,28,28)


##  then later to load:
loaded_model = DigitClassifier()
loaded_model.load_state_dict(torch.load(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_Digit_Recognizer\DigitClassifier_saved_model_10epochs.pt'))

test_preds = loaded_model(test_images_tensor)

softmax = torch.nn.Softmax(dim = -1)

test_preds = softmax(test_preds)

test_preds = torch.max(test_preds, -1)[1]


test_preds = test_preds.numpy()

test_preds = pd.DataFrame(test_preds, index = [i+1 for i in np.arange(len(test_preds))], columns = ['Label'])
test_preds.index.name = 'ImageId'

# test_preds.to_csv(r'C:\Users\M\OneDrive - Carleton University\Documents\my_stuff\Projects\Kaggle_Digit_Recognizer\predicted_digits_submission_3.csv')

print ('done')