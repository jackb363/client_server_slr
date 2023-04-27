import json

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from tensorflow import lite
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest
from django.core import serializers
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.http import HttpResponse
import os

# Construct the absolute path to the model file
MODEL_PATH = os.path.join('res', 'model.tflite')

interpreter = lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def index(request):
    template_path = os.path.join('../templates', 'index.html')
    return render(request, template_path)


@csrf_exempt
@require_POST
def detect_tflite(request):
    actions = np.array(['hello', 'iloveyou', 'thanks'])

    data = json.loads(request.body)
    sequence = np.array(data['data']).astype('float32').reshape(30, 1662)
    sequence = np.expand_dims(sequence, axis=0)
    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]

    header = str(actions[np.argmax(res)])
    print(header)
    return JsonResponse({'result': header})
