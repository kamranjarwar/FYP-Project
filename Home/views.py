from django.shortcuts import render, get_object_or_404, redirect
from django.http import StreamingHttpResponse
from django.conf import settings
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import authenticate, login as auth_login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from .models import Vehicle, UserProfile
from ultralytics import YOLO
from .forms import ImageUploadForm
import cv2
import easyocr
import re
import logging
import os



# --------------HOME (APP) START HERE------------#

def home(request):
    return render(request, 'index.html', {'is_authenticated': request.user.is_authenticated})  

# --------------HOME (APP) END HERE------------#



#-------------------------------------------Start-Validation--------------------------------------------#

# Registration View
def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        first_name = request.POST.get('first_name')  
        last_name = request.POST.get('last_name')  
        email = request.POST.get('email')
        password = request.POST.get('password')
        phone_number = request.POST.get('phone_number')
        gender = request.POST.get('gender')
        description = request.POST.get('description')  # Optional
        photo = request.FILES.get('photo')            # Optional

        # Check for existing username and email
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return redirect('register')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('register')

        # Basic validation
        if not all([username, first_name, last_name, email, password, phone_number, gender]):
            messages.error(request, "All fields except description and photo are required.")
            return redirect('register')

        # Create User
        user = User.objects.create_user(
            username=username,
            password=password,
            email=email,
            first_name=first_name,
            last_name=last_name
        )

        # Create UserProfile
        profile = UserProfile.objects.create(
            user=user,
            phone_number=phone_number,
            gender=gender,
            description=description
        )

        # Save Photo if Uploaded
        if photo:
            fs = FileSystemStorage()
            filename = fs.save(photo.name, photo)
            profile.photo = filename  # Save filename instead of URL
            profile.save()

        messages.success(request, "Registration successful! Please log in.")
        return redirect('login')

    return render(request, 'register.html')



def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            auth_login(request, user)  # Login the user
            return redirect('home')    # Redirect to the homepage
        else:
            # Display an error message when no account is found
            messages.error(request, "Sorry! No account found. Please check your username and password.")
            
    return render(request, 'login.html')



def logout_view(request):
    
    logout(request)
    
    return redirect('login')  


#-------------------------------------------End-Validation--------------------------------------------#




#-------------------------------------------Start-Profile--------------------------------------------#


@login_required(login_url='/login/')
def profile(request):
    user = request.user
    try:
        user_profile = UserProfile.objects.get(user=user)
    except UserProfile.DoesNotExist:
        user_profile = UserProfile(user=user)  # Create a user profile if it doesn't exist

    if request.method == 'POST' and 'edit_profile' in request.POST:
        # Update User model fields
        user.first_name = request.POST.get('first_name', user.first_name)
        user.last_name = request.POST.get('last_name', user.last_name)
        user.email = request.POST.get('email', user.email)

        # Update UserProfile model fields
        user_profile.phone_number = request.POST.get('phone_number', user_profile.phone_number)
        user_profile.description = request.POST.get('description', user_profile.description)
        user_profile.gender = request.POST.get('gender', user_profile.gender)

        # Handle photo upload
        if request.FILES.get('photo'):
            user_profile.photo = request.FILES['photo']

        # Save User and UserProfile models
        user.save()
        user_profile.save()

        messages.success(request, "Profile updated successfully!")
        return redirect('profile')  # Redirect back to the profile page

    if request.method == 'POST' and 'reset_password' in request.POST:
        # Handle password change
        password_form = PasswordChangeForm(request.user, request.POST)
        if password_form.is_valid():
            password_form.save()
            update_session_auth_hash(request, password_form.user)  # Keep the user logged in
            messages.success(request, "Password changed successfully!")
            return redirect('profile')  # Redirect to profile page after password change
        else:
            messages.error(request, "There was an error changing your password.")
            return render(request, 'profile.html', {'user': user, 'user_profile': user_profile, 'password_form': password_form})

    password_form = PasswordChangeForm(request.user)  # Create a new password change form

    return render(request, 'profile.html', {
        'user': user,
        'user_profile': user_profile,
        'password_form': password_form
    })

@login_required(login_url='/login/')
def delete_account(request):
    if request.method == 'POST':
        user = request.user
        user_profile = UserProfile.objects.filter(user=user).first()  # Get the profile object
        if user_profile:
            user_profile.delete()  # Delete the profile if it exists
        user.delete()  # Delete the user account
        messages.success(request, "Your account has been successfully deleted.")
        return redirect('home')  # Redirect to home after account deletion
    return redirect('profile')  # Redirect back to profile if the method isn't POST

#-------------------------------------------End-Profile--------------------------------------------#



#-------------------------------------------Start-Vehicles-Functions--------------------------------------------#

def vehicles_list(request):
    vehicles = Vehicle.objects.all()
    return render(request, 'details_admin.html', {'vehicles': vehicles})

def vehicle_details(request, vehicle_id):
    vehicle = get_object_or_404(Vehicle, id=vehicle_id)
    return render(request, 'result_admin.html', {'vehicle': vehicle})

@login_required(login_url='/login/')
def vehicle_details_user(request, id):  # Accept 'id' as a parameter
    vehicle = get_object_or_404(Vehicle, id=id)  # Get the vehicle by ID
    return render(request, 'details.html', {'vehicle': vehicle})


#--------------------------------------------End-Vehicles-Function--------------------------------------------#





#--------------------------------------------Start-Quick-Search--------------------------------------------#


@login_required(login_url='/login/')
def quick_search(request):
    context = {}
    
    if request.method == "GET":
        number_plate = request.GET.get('number_plate')
        
        if number_plate:
            try:
                # Query the database to find the vehicle with the number_plate
                vehicle = Vehicle.objects.get(number_plate=number_plate)
                
                # Add the vehicle details to the context
                context['vehicle'] = {
                    'number_plate': vehicle.number_plate,
                    'vehicle_type': vehicle.get_vehicle_type_display(),
                    'make': vehicle.make,
                    'model': vehicle.model,
                    'tax_status': vehicle.tax_status,
                    'id': vehicle.id,
                }
            except Vehicle.DoesNotExist:
                context['error_message'] = "No vehicle found with the provided number plate."
    
    return render(request, 'quick-search.html', context)


#--------------------------------------------End-Quick-Search--------------------------------------------#




#---------------------------------------------Start-live-Detection-----------------------------------------------------------#

# Setup logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model (adjust the path for your model)
model_path = 'Home/models/best.pt'  # Adjust path as needed
model = YOLO(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Plate text cleaning function
def clean_plate_text(text):
    """Clean up the OCR result text by removing spaces, special characters, and converting to uppercase."""
    text = re.sub(r'\W+', '', text)  # Remove all non-alphanumeric characters
    text = text.upper().strip()  # Convert to uppercase and strip leading/trailing whitespaces
    return text

# Store detected plate numbers globally (or in a session if persistent across views)
detected_numbers = []

# Function to process the image and detect plates
def process_image(img):
    global detected_numbers

    if img is None or img.size == 0:
        logger.error("Invalid frame encountered, skipping.")
        return img, None

    results = model(img, conf=0.5, iou=0.5)

    for result in results:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cropped_image = img[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue

            ocr_result = reader.readtext(cropped_image)
            result_text = ""
            max_confidence = 0.0

            for detection in ocr_result:
                text = detection[1]
                confidence = detection[2]
                result_text += text
                if confidence > max_confidence:
                    max_confidence = confidence

            result_text = clean_plate_text(result_text)
            logger.info(f"Detected plate text after cleaning: {result_text}")

            if max_confidence > 0.7:
                try:
                    vehicle = Vehicle.objects.get(number_plate__iexact=result_text)
                    logger.info(f"Vehicle found in database: {vehicle.id}")

                    # Add the plate to the detected numbers list (if not already present)
                    if result_text not in detected_numbers:
                        detected_numbers.append(result_text)

                    accuracy = int(max_confidence * 100)
                    display_text = f"{result_text} - {accuracy}%"
                    cv2.putText(img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                    # Send redirect signal along with the frame
                    return img, f'/details/{vehicle.id}/'

                except Vehicle.DoesNotExist:
                    logger.warning(f"Plate {result_text} not found in database.")
                    cv2.putText(img, "Plate Not Found in Database!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return img, None


# Function to generate the video stream for live feed
def generate_video_stream(request):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found or cannot be accessed")

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            break

        frame_counter += 1
        if frame_counter % 5 != 0:  # Skip frames if you don't need every frame
            continue

        processed_frame, redirect_url = process_image(frame)

        # Send the frame for live streaming
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        if not ret:
            logger.error("Failed to encode frame as JPEG.")
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        if redirect_url:
            # Send a redirect signal with the vehicle details URL
            yield (f'--redirect:{redirect_url}\r\n\r\n').encode('utf-8')
            break

    cap.release()

# View for live feed video streaming
def live_feed(request):
    return StreamingHttpResponse(generate_video_stream(request), content_type='multipart/x-mixed-replace; boundary=frame')



# View to show the detected license plate numbers
@login_required(login_url='/login/')
def detected_numbers_list(request):
    global detected_numbers

    # Optionally, check if the detected numbers exist in the database
    detected_vehicles = []
    for plate in detected_numbers:
        try:
            vehicle = Vehicle.objects.get(number_plate__iexact=plate)
            detected_vehicles.append(vehicle)
        except Vehicle.DoesNotExist:
            continue  # Skip if no matching vehicle is found

    # Render the list page with detected vehicles
    return render(request, 'live-detection.html', {'detected_vehicles': detected_vehicles})



#---------------------------------------------End-live-Detection-----------------------------------------------------------#




#----------------------------------------Start-Image-Detection---------------------------------------------------#


# Your YOLO and OCR model setup
yolo_model = YOLO("home/yolo/yolov8m.pt")  # Path to the YOLOv8 vehicle detection model
license_plate_model = YOLO("home/models/best.pt")  # Path to the YOLOv8 license plate detection model

# Define vehicle classes (Car, Truck, Bike, Bus)
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def process_image_and_detect(image_path):
    """
    Processes an image to detect vehicles and license numbers.
    Arguments:
        image_path: Path to the image file.
    Returns:
        detected_plates: List of dictionaries with license plate text, vehicle type, and matched vehicle details.
    """
    # Load image
    image = cv2.imread(image_path)
    detected_plates = []  # List to store detected plate info

    # Run vehicle detection using YOLO
    vehicle_results = yolo_model(image)
    for result in vehicle_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes for detected vehicles
        class_ids = result.boxes.cls.cpu().numpy()  # Get the class IDs for detected vehicles

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
            vehicle_label = VEHICLE_CLASSES.get(int(class_id), "Unknown")  # Get the vehicle type

            # Crop the detected vehicle region
            vehicle_crop = image[y1:y2, x1:x2]

            # Detect license plates using the license plate detection model
            plate_results = license_plate_model(vehicle_crop)
            for plate_result in plate_results:
                plate_boxes = plate_result.boxes.xyxy.cpu().numpy()  # Get bounding boxes for number plates
                for px1, py1, px2, py2 in plate_boxes:
                    px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])  # Plate box coordinates
                    plate_crop = vehicle_crop[py1:py2, px1:px2]  # Crop the number plate

                    # Convert plate region to grayscale for OCR
                    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    
                    # Use EasyOCR to extract text from the number plate
                    ocr_results = reader.readtext(plate_gray)

                    for (_, text, _) in ocr_results:
                        # Check if the license plate exists in the database
                        vehicle = Vehicle.objects.filter(number_plate__iexact=text).first()

                        # Only show the license number and vehicle type
                        detected_plates.append({
                            "license_number": text,
                            "vehicle_type": vehicle_label,
                            "vehicle_exists": bool(vehicle),  # True if the vehicle exists in the database
                            "vehicle_details_url": f"/details/{vehicle.id}/" if vehicle else None
                        })

    return detected_plates

def image_detection(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            
            # Get the image path relative to the media directory
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)

            # Save the uploaded image
            with open(image_path, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Process the image
            detected_plates = process_image_and_detect(image_path)

            # Render results to the template (showing license number, vehicle type, and button)
            return render(request, "image-detection.html", {
                "plates": detected_plates,
                "image": settings.MEDIA_URL + image.name  # Use MEDIA_URL to display image
            })

    else:
        form = ImageUploadForm()

    return render(request, "image-detection.html", {"form": form})


#----------------------------------------End-Image-Detection---------------------------------------------------#
