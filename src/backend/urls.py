from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

import backend.views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", backend.views.scoreImagePage, name="scoreImagePage"),
    path("predictImage", backend.views.predictImage, name="predictImage"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
