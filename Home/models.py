from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    photo = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    phone_number = models.CharField(max_length=15)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    
  





class Vehicle(models.Model):

    VEHICLE_TYPES = [
        ('car', 'Car'),
        ('bus', 'Bus'),
        ('truck', 'Truck'),
        ('motorcycle', 'Motorcycle'),
    ]
    
    TAX_STATUS = [
        (True, 'Paid'),
        (False, 'Unpaid'),
    ]
    
    number_plate = models.CharField(max_length=20, unique=True)
    vehicle_type = models.CharField(choices=VEHICLE_TYPES, max_length=20)
    make = models.CharField(max_length=50)
    model = models.CharField(max_length=50)
    color = models.CharField(max_length=50)
    variant = models.CharField(max_length=50, blank=True, null=True)
    country = models.CharField(max_length=50)
    registration_date = models.DateField()
    tax_status = models.BooleanField(choices=TAX_STATUS, default=False) 
    registered_owner = models.CharField(max_length=100)
    owner_address = models.TextField()  
    owner_fathers_name = models.CharField(max_length=100) 
    is_blacklisted = models.BooleanField(default=False)

    
    def __str__(self):
        return self.number_plate

class DetectionLog(models.Model):
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, null=True, blank=True)
    detected_plate = models.CharField(max_length=20)
    detection_time = models.DateTimeField(auto_now_add=True)
    is_matched = models.BooleanField(default=False)

    def __str__(self):
        return f"Detection: {self.detected_plate} at {self.detection_time}"