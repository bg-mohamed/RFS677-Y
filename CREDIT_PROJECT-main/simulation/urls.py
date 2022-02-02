from django.urls import path, include
from . import views
app_name = 'simulation'

urlpatterns = [
    path('', views.simulation),
    path('/simulation', views.simulation, name="sim"),
    path('/applicant', views.applicant, name="applicant"),
    path('/charge', views.charge, name="charge"),
    path('/pro_situation', views.pro_situation, name="pro_situation"),
    #path('/project', views.project, name="project"),
]
