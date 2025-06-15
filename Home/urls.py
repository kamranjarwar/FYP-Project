from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),  
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile, name='profile'),
    path('delete-account/', views.delete_account, name='delete-account'),
    path('quick-search/', views.quick_search, name='quick-search'),
    path('live_feed/', views.live_feed, name='live_feed'),
    path('live-detection/', views.detected_numbers_list, name='live-detection'),
    path('video_stream/', views.generate_video_stream, name='video_stream'),
    path('image-detection/',views.image_detection, name='image-detection'),
    path('vehicles/', views.vehicles_list, name='vehicles_list'),
    path('vehicles/<int:vehicle_id>/', views.vehicle_details, name='vehicle_details'),
    path('details/<int:id>/', views.vehicle_details_user, name='details'),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

