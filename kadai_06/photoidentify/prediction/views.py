from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        # GETリクエストによるアクセス時の処理を記述
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        # POSTリクエストによるアクセス時の処理を記述
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            #img_array = img_array/255
            #model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'model.h5')
            #model = load_model(model_path)
            model = VGG16(weights='imagenet')
            result = model.predict(img_array)
            #print(result)
            prediction = decode_predictions(result)
            #print(prediction)
            #if result[0][0] > result[0][1]:
            #    prediction = '猫'
            #else:
            #    prediction = '犬'
            img_data = request.POST.get('img_data')
            return render(request, 'home_table.html', {'form': form, 'prediction': prediction[0], 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})