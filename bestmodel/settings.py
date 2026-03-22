"""
Django settings for bestmodel project.
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent


# ── Security ──────────────────────────────────────────────────────────────────

SECRET_KEY = 'django-insecure-38^xf0*fg-jt^l=q+szkx&z$k$v--n+17jc*t)&e47-3p-$y*r'

DEBUG = True

ALLOWED_HOSTS = []


# ── Applications ──────────────────────────────────────────────────────────────

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'predictor',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CORS_ALLOW_ALL_ORIGINS = True

ROOT_URLCONF = 'bestmodel.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'bestmodel.wsgi.application'


# ── Database ──────────────────────────────────────────────────────────────────

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# ── Password validation ───────────────────────────────────────────────────────

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# ── Internationalisation ──────────────────────────────────────────────────────

LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'UTC'
USE_I18N      = True
USE_TZ        = True


# ── Static and media files ────────────────────────────────────────────────────

STATIC_URL = 'static/'

# Grad-CAM saved images are served from /media/
MEDIA_URL  = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')


# ── FortifyAI model configuration ─────────────────────────────────────────────
#
# CHANGE ONLY THIS LINE per machine — everything else stays the same.
#
# Your path (Developer 1):
FORTIFYAI_CHECKPOINT = r"E:\Fortifyai\checkpoints\best_model.pth"
#
# Partner's path (Developer 2) — they set this on their own machine:
# FORTIFYAI_CHECKPOINT = r"C:\Users\partner\path\to\checkpoints\best_model.pth"
#
# Grad-CAM outputs are saved here with timestamp filenames
GRADCAM_OUTPUT_DIR = os.path.join(MEDIA_ROOT, 'gradcam_outputs')


# ── Misc ──────────────────────────────────────────────────────────────────────

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
