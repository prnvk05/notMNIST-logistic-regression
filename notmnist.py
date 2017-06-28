# modules to import
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image, ImageDraw
from scipy.spatial.distance import cosine
data_root = '.'


num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract('notMNIST_large.tar.gz')
test_folders = maybe_extract('notMNIST_small.tar.gz')

# To check if the dataset has been downloaded properly, display an image from the data set
width = 28
height = 28
channels = 4
image = Image.open("notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png")
image.show()
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

# check if pickling went well



def check_imgs(dataset):
  ''' To verify the modified dataset'''
  size={}
  if isinstance(dataset,np.ndarray)==True:
      
      for i in range(10):
          disp_slice= train_dataset[i,:,:]
          plt.figure()
          plt.imshow(disp_slice)  # display it
          plt.show()
      return 0
    
  else:
      for i in dataset:
          pick_sample= i
          with open(pick_sample,'rb') as f:
              disp_unpack=pickle.load(f)
              size[i]={len(disp_unpack)}
              disp_index=np.random.randint(len(disp_unpack))
              disp_slice= disp_unpack[disp_index,:,:]
              plt.figure()
              plt.imshow(disp_slice)  # display it
              plt.show()

      return size
  
# verify that the dataset is balanced
x=check_imgs(test_datasets)
for k,v in x.items():
  print(k," : " ,v)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
_=check_imgs(train_dataset)
_=check_imgs(test_dataset)
_=check_imgs(valid_dataset)

def find_overlap(set1,set1_labels, set2, set2_labels):

    '''To find overlap between two datasets'''
    count= 0
    for i in range(len(set1)):

        for j in range(i+1,len(set2)):

            if set1_labels[i]==set2_labels[j]:

                if cosine(np.hstack(set1[i]),np.hstack(set2[j]))<0.01:

                    count+= 1

    

    print("Number of similar images ",count)

    

#find_overlap(train_dataset, train_labels, test_dataset, test_labels)

#find_overlap(train_dataset, train_labels, valid_dataset, valid_labels)
for train_examples_count in [50,100,1000,5000]:
    #Train
    logit =LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000)
    small_train_data = train_dataset[:train_examples_count]
    small_train_lbl = train_labels[:train_examples_count]
    #Each Letter is a 28*28 pixel image. Convert 28*28 matrix to a 1*784 array. Do this for all images
    X_train = small_train_data.reshape(len(small_train_data),len(small_train_data[0])*len(small_train_data[0]))
    logit.fit(X_train,small_train_lbl)
    print(logit.score(X_train,small_train_lbl))

#prediction

X_test = test_dataset.reshape(len(test_dataset),len(test_dataset[0])*len(test_dataset[0]))
pred_lst = [(logit.predict(row.reshape(1,-1)))[0] for row in X_test]

alpha=['a','b','c','d','e','f','g','h','i','j']

for i in range(len(pred_lst)):
    print(alpha[pred_lst[i]]," : ")
    disp_slice= test_dataset[i,:,:]
    plt.figure()
    plt.imshow(disp_slice)  # display it
    plt.show()




