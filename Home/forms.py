from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()


class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label='Upload Video',
        widget=forms.ClearableFileInput(attrs={'accept': 'video/*'})
    )
