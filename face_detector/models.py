from django.db import models
# Create your models here.
''''''
import uuid
from django.contrib.postgres.fields import ArrayField

class Shopper(models.Model):
    #id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    id = models.CharField(primary_key=True,max_length=200,default='default')
    url = models.CharField(max_length=200,default='defect')
    feature = ArrayField(models.FloatField(),default=[0.1,0.2])

#class Post(models.Model):
#    name = models.CharField(max_length=200)
#    tags = ArrayField(models.CharField(max_length=200), blank=True)
