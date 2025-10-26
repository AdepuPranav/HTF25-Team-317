from django.shortcuts import render
from django.conf import settings

def index(request):
    # Backend base URL (FastAPI). In dev, it's localhost:8000
    backend_base = request.GET.get('backend', 'http://127.0.0.1:8000')
    return render(request, 'maps/index.html', { 'BACKEND_BASE': backend_base })
