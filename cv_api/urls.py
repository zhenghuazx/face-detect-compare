from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
                       # Examples:

                       url(r'^face_detection/detect/$', 'face_detector.views.detect'),
                       url(r'^face_detection/getId/$', 'face_detector.views.getId'),
                       # url(r'^$', 'cv_api.cv_api.views.home', name='home'),
                       # url(r'^blog/', include('blog.urls')),

                       url(r'^admin/', include(admin.site.urls)),
                       )