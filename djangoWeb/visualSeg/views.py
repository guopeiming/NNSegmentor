from django.shortcuts import render
from django.http import HttpResponse
import json
from django.conf import settings
import time


def index(request):
    return render(request, 'visualSeg/index.html')


def seg(request):
    inst = request.GET['content']
    print(inst)
    start = time.time()
    result = settings.SEG(inst)
    end = time.time()-start
    temp = result.argmax(1).cpu().tolist()
    res = ''
    for i in range(len(temp)):
        res += (' ' + inst[i]) if temp[i] == 1 else inst[i]
    return HttpResponse(json.dumps({'result': res, 'BERT': 6.0732784648, 'NNSegmentor': end}), content_type='application/json')
