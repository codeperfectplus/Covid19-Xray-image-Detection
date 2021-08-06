from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50

test_data_dir = Path("Covid19-dataset/test")
data_gen = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

test_data = data_gen.flow_from_directory(test_data_dir,
                                         target_size=(224, 224),
                                         color_mode='rgb',
                                         shuffle=True)


model = load_model("tmp/saved_model.h5")

evaluation = model.evaluate(test_data)
print(evaluation)
