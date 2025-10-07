from django.db import models

# Create your models here.

from django.contrib.auth.models import AbstractUser
from django.db import models



# class CustomUser(AbstractUser):
#     # your custom fields here
#     pass

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username
