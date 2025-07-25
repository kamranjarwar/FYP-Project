# Generated by Django 5.1.5 on 2025-02-04 21:31

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Vehicle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('number_plate', models.CharField(max_length=20, unique=True)),
                ('vehicle_type', models.CharField(choices=[('car', 'Car'), ('bus', 'Bus'), ('truck', 'Truck'), ('motorcycle', 'Motorcycle')], max_length=20)),
                ('make', models.CharField(max_length=50)),
                ('model', models.CharField(max_length=50)),
                ('variant', models.CharField(blank=True, max_length=50, null=True)),
                ('country', models.CharField(max_length=50)),
                ('registration_date', models.DateField()),
                ('tax_status', models.BooleanField(choices=[(True, 'Paid'), (False, 'Unpaid')], default=False)),
                ('registered_owner', models.CharField(max_length=100)),
                ('owner_address', models.TextField()),
                ('owner_fathers_name', models.CharField(max_length=100)),
                ('is_blacklisted', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='DetectionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('detected_plate', models.CharField(max_length=20)),
                ('detection_time', models.DateTimeField(auto_now_add=True)),
                ('is_matched', models.BooleanField(default=False)),
                ('vehicle', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='Home.vehicle')),
            ],
        ),
    ]
