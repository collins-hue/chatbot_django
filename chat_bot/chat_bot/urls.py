
from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('', include('chatbot.urls', namespace='chatbot')),
    path('admin/', admin.site.urls),
]
