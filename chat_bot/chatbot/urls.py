from django.urls import path
from . import views

app_name = 'chatbot'
urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('', views.chat_view, name='chat_view'),
]

