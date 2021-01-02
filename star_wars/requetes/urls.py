from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('description', views.description, name='description'),
    path('correlation', views.correlation, name='correlation'),
    path('classification', views.classification, name='classification'),
]
