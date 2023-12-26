from keras.applications.vgg16 import VGG16
from keras.utils import plot_model
model = VGG16()
from keras.applications.vgg16 import VGG16
model = VGG16()
model.summary()
plot_model(model, to_file='vgg.png')