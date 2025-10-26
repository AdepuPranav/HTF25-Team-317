from django.shortcuts import render

def dashboard(request):
    # Renders dashboard.html template
    return render(request, 'dashboard.html', {})
