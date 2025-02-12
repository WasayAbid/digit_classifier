from django.contrib import admin
from django.urls import path
from .views import home, classify_digit

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),  # Home route for index.html
    path('predict/', classify_digit, name='predict-digit'),  # API route
]
