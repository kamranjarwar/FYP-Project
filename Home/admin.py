from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from .models import Vehicle, DetectionLog, UserProfile



# Customize the Admin Panel Titles
admin.site.site_header = "PlateCheck Admin Dashboard"          # Header on the login page
admin.site.site_title = "PlateCheck Admin Portal"             # Title in the browser tab
admin.site.index_title = "Welcome to PlateCheck Admin Panel"  # Title on the admin homepage


class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'gender', 'description')  # Fields to show in list view
    search_fields = ('user__username', 'first_name', 'last_name', 'email')  # Search fields
    list_filter = ('gender', 'user__is_active')  # Filter options
    ordering = ('user__username',)  # Default ordering of items

admin.site.register(UserProfile, UserProfileAdmin)




@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ('number_plate', 'vehicle_type', 'tax_status_display', 'owner_address', 'owner_fathers_name', 'view_details_button')

    def tax_status_display(self, obj):
        if obj.tax_status:
            return format_html('<span style="color:green;">Paid</span>')
        else:
            return format_html('<span style="color:red;">Unpaid</span>')
    
    tax_status_display.short_description = 'Tax Status'

    def view_details_button(self, obj):
        url = reverse('vehicle_details', args=[obj.id])
        return format_html('<a class="button" href="{}">View Details</a>', url)
    
    view_details_button.short_description = 'Details'
    view_details_button.allow_tags = True

@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ('detected_plate', 'detection_time', 'is_matched')


