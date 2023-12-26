from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from os import listdir, path


#
# image = load_img('../data/Flickr8k/Images/990890291_afc72be141.jpg')
#
# image = img_to_array(image)
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# image = load_img('990890291_afc72be141.jpg', target_size=(224, 224))


def load_photos(directory):
    images = dict()
    for name in listdir(directory):
        filename = path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get image id
        image_id = name.split('.')[0]
        images[image_id] = image
    return images
directory = '../data/Flickr8k/Images'
images = load_photos(directory)
print('Loaded Images: %d' % len(images))