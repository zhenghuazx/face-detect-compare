# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Shopper',
            fields=[
                ('id', models.CharField(default=b'default', max_length=200, serialize=False, primary_key=True)),
                ('url', models.CharField(default=b'defect', max_length=200)),
                ('feature', django.contrib.postgres.fields.ArrayField(default=[0.1, 0.2], base_field=models.FloatField(), size=None)),
            ],
        ),
    ]
