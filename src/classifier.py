import os
from pathlib import Path
import json
import time
import logging
import re
import modal


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Override to pick up changes from setup_drive_folders
    print("Loaded environment variables from .env file.")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading.")




#initialize secrets 
_default_local_downloads = Path.home() / "Downloads" / "All_Downloads"
LOCAL_DOWNLOADS_DIR = Path(os.environ.get("LOCAL_DOWNLOADS_DIR", str(_default_local_downloads)))

CONTAINER_DOWNLOADS_DIR = Path(os.environ.get("CONTAINER_DOWNLOADS_DIR", "/Downloads/"))

SECRET_NAME = os.environ.get("MODAL_SECRET_NAME", "google_drive_secret")

ROOT_FOLDER_NAME = os.environ.get("GDRIVE_ROOT_FOLDER_NAME", "Classified_Posts")

GPU_CONFIG = os.environ.get("MODAL_GPU_CONFIG", "L40S")

PYTHON_VERSION = os.environ.get("MODAL_PYTHON_VERSION", "3.10")

GDRIVE_PARENT_FOLDER_ID = os.environ.get("GDRIVE_PARENT_FOLDER_ID")

# --- Logging Setup ---
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
# Basic config for local execution, Modal might override format in containers
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger


if not GDRIVE_PARENT_FOLDER_ID:
    if modal.is_local(): #we've gotta set this up locally
        logger.info("Running locally, GDRIVE_PARENT_FOLDER_ID is not required.")
        raise ValueError("GDRIVE_PARENT_FOLDER_ID environment variable is not set. Halting execution. Please init the environment file")


# --- Modal Image Definition ---
# Defines the container environment
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "ftfy",
        "regex",
        "tqdm",
        "Pillow",
        "git+https://github.com/openai/CLIP.git",
        "transformers",
        "google-api-python-client",
        "google-auth-oauthlib",
        "google-auth-httplib2",
        "python-dotenv", # Needed for .env loading if used inside container (less common)
    )
    .add_local_dir(str(LOCAL_DOWNLOADS_DIR), str(CONTAINER_DOWNLOADS_DIR))
)

# Conditionally add probe weights to container image if they exist
_probe_weights_path = Path(__file__).resolve().parent / "probe_weights.pt"
if _probe_weights_path.exists():
    image = image.add_local_file(str(_probe_weights_path), "/probe/probe_weights.pt")
    logger.info("Probe weights found, adding to container image.")
else:
    logger.info("No probe weights found at src/probe_weights.pt — will use zero-shot fallback.")

# Create a Modal App instance
app = modal.App(f"clip-classifier-{ROOT_FOLDER_NAME.lower().replace(' ', '-')}", image=image)



def create_drive_folder(service, folder_name, parent_folder_id=None):
    """Creates a new folder in Google Drive or returns its ID if it already exists."""
    from googleapiclient.errors import HttpError

    # Escape single quotes in folder names for the query
    safe_folder_name = folder_name.replace("'", "\\'")

    query = (
        f"mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{safe_folder_name}' "
        f"and trashed = false"
    )
    if parent_folder_id:
        query += f" and '{parent_folder_id}' in parents"
    else:
        # If checking in root (no parent specified), ensure it's directly in 'root'
        query += " and 'root' in parents"

    try:
        response = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
        folders = response.get('files', [])
        if folders:
            folder_id = folders[0]['id']
            logger.debug(f"Folder '{folder_name}' already exists with ID: {folder_id} under parent {parent_folder_id or 'root'}")
            return (False, folder_id)
    except HttpError as error:
        logger.error(f"An error occurred checking for folder '{folder_name}': {error}")
        return None # Indicate failure

    # Folder does not exist, create it
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]

    try:
        file = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = file.get('id')
        logger.info(f"Created folder '{folder_name}' with ID: {folder_id} under parent {parent_folder_id or 'root'}")
        return (True, folder_id)
    except HttpError as error:
        logger.error(f"An error occurred creating folder '{folder_name}': {error}")
        return None # Indicate failure


def upload_to_drive(service, folder_id, file_path):
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    import mimetypes # Use standard library for mime types

    file_name = file_path.name
    # Guess mime type using standard library, provide default
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        # Handle special cases or provide a default
        if file_name.endswith(".json.xz"):
            mime_type = "application/x-xz"
        else:
            mime_type = "application/octet-stream" # Generic binary

    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

    try:
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        logger.info(f"Uploaded '{file_name}' to folder ID '{folder_id}', file ID '{file.get('id')}'")
        return True
    except Exception as e:
        logger.error(f"An error occurred while uploading '{file_name}': {e}")
        return False 

@app.cls(
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name(SECRET_NAME), modal.Secret.from_dotenv()],
    timeout=1800, # 30 minutes timeout per container
    max_containers=10, 
    min_containers=2
    
)
class Classifier:
        
    @modal.enter()
    def start(self): 
        
        self.classification_folder_id = os.environ.get("CLASSIFICATION_FOLDER_ID")
        if not self.classification_folder_id:
            raise ValueError("CLASSIFICATION_FOLDER_ID environment variable not set in container. Cannot proceed.")
        logger.info(f"Using classification folder ID: {self.classification_folder_id}")
        
        self.category_folders = { "opioid_related": os.environ.get("OPIOID_RELATED_FOLDER_ID"),
                                 "neutral_content": os.environ.get("NEUTRAL_CONTENT_FOLDER_ID"),
                                 "error":os.environ.get("ERROR_FOLDER_ID") }
        if not self.category_folders:
            raise ValueError("Category folder IDs not set in environment. Cannot proceed.")
        
        # Confidence threshold configuration (0.0 to 1.0, where higher = more confident required)
        # If confidence below threshold, default to neutral_content
        self.image_confidence_threshold = float(os.environ.get("CLIP_CONFIDENCE_THRESHOLD", "0.0"))
        if not (0.0 <= self.image_confidence_threshold <= 1.0):
            raise ValueError(f"CLIP_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {self.image_confidence_threshold}")
        
        # Text classification configuration
        self.text_confidence_threshold = float(os.environ.get("TEXT_CONFIDENCE_THRESHOLD", "0.6"))
        if not (0.0 <= self.text_confidence_threshold <= 1.0):
            raise ValueError(f"TEXT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {self.text_confidence_threshold}")
        
        self.text_classifier_model = os.environ.get("TEXT_CLASSIFIER_MODEL", "valhalla/distilbart-mnli-12-3")
        
        # Temperature scaling for logits (higher = peakier probabilities)
        self.temperature = float(os.environ.get("CLIP_TEMPERATURE", "100.0"))
        if self.temperature <= 0:
            raise ValueError(f"CLIP_TEMPERATURE must be positive, got {self.temperature}")
        
        self.CATEGORIES = {
            "opioid_related": [
                "heroin injection",
                "fentanyl pills",
                "oxycodone pills",
                "opioid overdose",
                "prescription opioid abuse",
                "illegal opioid sales",
                "opioid manufacturing",
                "counterfeit pills",
                "opioid crisis",
                "IV drug use",
                "black tar heroin",
                "opioid addiction",
                "opioid withdrawal",
                "needle exchange",
                "opioid death",
                "naloxone administration",
                "opioid street price",
                "opioid trafficking",
                "pill press",
                "substance-induced fatality",
                "drug paraphernalia",
                "syringes",
                "drug injection",
                "controlled substance",
                "painkillers",
                "addiction recovery",
                "rehabilitation center",
                "sobriety support",
                "harm reduction",
                "safe injection",
                "overdose prevention",
                "drug education",
                "addiction treatment",
                "recovery meeting",
                "12 step program",
                "relapse prevention",
                "drug testing",
                "clean needle program",
                "naloxone kit",
                "medication-assisted treatment",
                "sober living",
                "counseling for addiction",
                "support groups",
                "detoxification",
                "intervention",
            ],

            "neutral_content": [
                "a natural landscape",
                "food and drink",
                "people socializing",
                "pets and animals",
                "daily activities",
                "travel photo",
                "sports and recreation",
                "technology",
                "art and creativity",
                "fashion and style",
                "vehicles and transportation",
                "home decoration",
                "office environment",
                "a cooking recipe",
                "fitness exercise",
                "a news event",
                "political discussion",
                "educational material",
                "a motivational quote",
                "family gathering",
                "holiday celebration",
                "nature photography",
                "architecture photo",
                "abstract art", 
                "advertisement",
                "promotional material",
                "a photo of everyday items",
                "a photo of normal activities",
                "a photo of common household items",
                "a photo of regular food",
                "a photo of casual social interaction",
                "a photo of pets playing",
                "a photo of outdoor scenery",
                "a photo of buildings and architecture",
                "a photo of clothing and fashion",
                "a photo of electronic devices",
                "a photo of vehicles on the street",
                "a photo of people exercising",
                "a photo of family events",
                "a photo of holiday decorations",
                "a photo of artwork and paintings",
                "a photo of sports equipment",
                "a photo of cooking ingredients",
                "a photo of nature and wildlife",
                "a photo of urban scenes",
                "a photo of office supplies",
                "a photo of home furnishings",
                "a photo of recreational activities",
                "a photo of social media content",
                "a photo of memes and humor",
                "a photo of personal moments",
                "a photo of cultural events",
                "a photo of entertainment",
                "a photo of hobbies and interests",
            ]
        }

        # Generate prompts and map them back to categories *once*
        self.ALL_PROMPTS = []
        self.PROMPT_TO_CATEGORY_MAP = []
        for category, prompts_list in self.CATEGORIES.items():
            formatted_prompts = [f"a photo of {prompt}" for prompt in prompts_list]
            self.ALL_PROMPTS.extend(formatted_prompts)
            self.PROMPT_TO_CATEGORY_MAP.extend([category] * len(prompts_list))

      
        logger.info(f"Initializing container on Python {PYTHON_VERSION} with GPU {GPU_CONFIG}...")

        import torch
        import clip
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        try:

            self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
            logger.info("CLIP model ViT-L/14 loaded.")
        except Exception as e:
            logger.exception("Failed to load CLIP model!")
            raise  # Critical error, stop container initialization

        # Load linear probe if available
        self.use_probe = os.environ.get("USE_LINEAR_PROBE", "true").lower() == "true"
        self.probe = None
        self.probe_threshold = 0.5
        if self.use_probe:
            probe_path = Path("/probe/probe_weights_2.pt")
            if probe_path.exists():
                try:
                    probe_data = torch.load(probe_path, weights_only=False, map_location=self.device)
                    self.probe = torch.nn.Linear(768, 1).to(self.device)
                    self.probe.load_state_dict(probe_data["weights"])
                    self.probe.eval()
                    self.probe_threshold = probe_data.get("threshold", 0.5)
                    logger.info(f"Linear probe loaded (threshold={self.probe_threshold:.3f})")
                except Exception as e:
                    logger.warning(f"Failed to load probe weights: {e}. Falling back to zero-shot.")
                    self.probe = None
                    self.use_probe = False
            else:
                logger.info("Probe weights not found at /probe/probe_weights.pt. Using zero-shot fallback.")
                self.use_probe = False

        # Pre-tokenize and encode text prompts
        try:
            with torch.no_grad():
                text_inputs = clip.tokenize(self.ALL_PROMPTS).to(self.device)
                self.text_features = self.model.encode_text(text_inputs)
                # Normalize text features once for efficient comparison later
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            logger.info(f"Encoded {len(self.ALL_PROMPTS)} text prompts.")
        except Exception as e:
             logger.exception("Failed to encode text prompts!")
             raise # Critical error

        # Load zero-shot text classifier
        try:
            from transformers import pipeline
            self.text_classifier = pipeline(
                "zero-shot-classification",
                model=self.text_classifier_model,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Zero-shot text classifier '{self.text_classifier_model}' loaded.")
        except Exception as e:
            logger.exception(f"Failed to load text classifier '{self.text_classifier_model}'!")
            raise  # Critical error, stop container initialization

        # Build Google Drive API client using service account from mounted secret
        try:
            service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
            creds = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=["https://www.googleapis.com/auth/drive.file"] # Scope for creating files/folders
            )
            self.drive_service = build("drive", "v3", credentials=creds)
            logger.info("Google Drive service client built successfully.")
        except KeyError:
            logger.error(f"SECRET ERROR: 'SERVICE_ACCOUNT_JSON' not found in environment. Ensure Modal secret '{SECRET_NAME}' is populated correctly.")
            raise # Critical error, cannot proceed without Drive access
        except Exception as e:
            logger.exception("Failed to build Google Drive service client!")
            raise # Critical error

    def _read_caption(self, image_path: Path) -> str | None:
        """
        Reads caption text from corresponding .txt file and preprocesses it.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed caption text with hashtags removed, or None if missing/invalid
        """
        try:
            # Find corresponding .txt file (same stem as image filename)
            txt_path = image_path.with_suffix('.txt')
            
            if not txt_path.exists():
                logger.debug(f"No caption file found for {image_path.name}")
                return None
            
            # Read caption text
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read()
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                try:
                    with open(txt_path, 'r', encoding='latin-1') as f:
                        caption_text = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read caption file {txt_path}: {e}")
                    return None
            
            # Preprocess caption: Remove all hashtags (words starting with #)
            # Use regex to remove hashtags and clean up extra whitespace
            caption_text = re.sub(r'#\w+', '', caption_text)
            caption_text = re.sub(r'\s+', ' ', caption_text)  # Normalize whitespace
            caption_text = caption_text.strip()
            
            # Handle empty/short captions
            if not caption_text or len(caption_text.strip()) < 3:
                logger.debug(f"Caption too short or empty after preprocessing for {image_path.name}")
                return None
            
            return caption_text
            
        except Exception as e:
            logger.warning(f"Error reading caption for {image_path.name}: {e}")
            return None

    def _classify_text(self, caption: str) -> tuple[dict[str, float], float]:
        """
        Classifies caption text using zero-shot classification.
        
        Args:
            caption: Caption text to classify
            
        Returns:
            Tuple of (category_probabilities_dict, confidence_score)
            confidence_score is the difference between top 2 probabilities
        """
        try:
        
            # Use zero-shot classifier with category labels
            category_labels = ["opioid_related", "neutral_content"]
            result = self.text_classifier(caption, category_labels)
            
            # Extract probabilities from result
            # Result format: {"labels": [...], "scores": [...]}
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            
            # Create dictionary mapping categories to probabilities
            category_probs = {label: score for label, score in zip(labels, scores)}
            
            # Ensure both categories are present (default to 0.0 if missing)
            category_probs.setdefault("opioid_related", 0.0)
            category_probs.setdefault("neutral_content", 0.0)
            
            # Calculate confidence as the difference between top 2 probabilities
            sorted_probs = sorted(category_probs.values(), reverse=True)
            confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else sorted_probs[0]
            
            # Log for debugging
            logger.debug(f"Text classification result - Labels: {labels}, Scores: {scores}, Confidence (difference): {confidence:.4f}")
            
            return (category_probs, confidence)
            
        except Exception as e:
            logger.warning(f"Error classifying text: {e}")
            # Return low confidence to trigger fallback to image classification
            return ({"opioid_related": 0.5, "neutral_content": 0.5}, 0.5)

    def _analyze_image(self, image_path: Path):
        """
        Analyzes a single image using fallback logic: text classification if confident, otherwise image classification.

        Returns:
            Tuple of (category, classification_method) where method is "text", "image", or "error".
        """
        import torch
        from PIL import Image, UnidentifiedImageError

        analyze_start_time = time.time()
        try:
            # Extract caption using _read_caption()
            caption = self._read_caption(image_path)
            
            # Try text classification if caption exists
            if caption:
                text_probs, text_confidence = self._classify_text(caption)
                
                # Log the comparison for debugging
                logger.info(f"Text classification for {image_path.name}: confidence={text_confidence:.4f}, threshold={self.text_confidence_threshold:.4f}, will_fallback={text_confidence < self.text_confidence_threshold}")
                
                # If text confidence >= TEXT_CONFIDENCE_THRESHOLD: use text classification result
                if text_confidence >= self.text_confidence_threshold:
                    best_category = max(text_probs, key=text_probs.get)
                    duration = time.time() - analyze_start_time
                    logger.info(f"Item {image_path.name} classified via TEXT as: {best_category} in {duration:.2f}s. Confidence: {text_confidence:.3f}. (Scores: { {k: f'{v:.3f}' for k, v in text_probs.items()} })")
                    return (best_category, "text")
                else:
                    # Text confidence < threshold: fallback to image classification
                    logger.info(f"Text confidence ({text_confidence:.4f}) < threshold ({self.text_confidence_threshold:.4f}) for {image_path.name}, falling back to image classification")
            
            # Fallback to image classification (either no caption or low text confidence)
            # Open and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            if self.use_probe and self.probe is not None:
                # Linear probe path
                with torch.no_grad():
                    logit = self.probe(image_features.float()).squeeze()
                    prob_opioid = torch.sigmoid(logit).item()
                best_category = "opioid_related" if prob_opioid >= self.probe_threshold else "neutral_content"
                category_scores = {"opioid_related": prob_opioid, "neutral_content": 1 - prob_opioid}
                confidence = abs(prob_opioid - self.probe_threshold)

                duration = time.time() - analyze_start_time
                logger.info(f"Item {image_path.name} classified via PROBE as: {best_category} in {duration:.2f}s. P(opioid)={prob_opioid:.3f} threshold={self.probe_threshold:.3f}")
                return (best_category, "image")
            else:
                # Zero-shot prompt matching fallback
                with torch.no_grad():
                    logits = (self.temperature * image_features @ self.text_features.T).squeeze(0)

                category_logits_list = []
                category_order = list(self.CATEGORIES.keys())
                for category in category_order:
                    category_indices = [i for i, cat in enumerate(self.PROMPT_TO_CATEGORY_MAP) if cat == category]
                    if category_indices:
                        indices_tensor = torch.tensor(category_indices, device=self.device)
                        category_logit = logits[indices_tensor].mean()
                        category_logits_list.append(category_logit)
                    else:
                        category_logits_list.append(torch.tensor(float('-inf'), device=self.device))

                category_logits_tensor = torch.stack(category_logits_list)
                category_probs = category_logits_tensor.softmax(dim=-1).cpu().numpy()
                category_scores = {cat: prob for cat, prob in zip(category_order, category_probs)}

                sorted_probs = sorted(category_probs, reverse=True)
                confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else sorted_probs[0]

                if not category_scores:
                    best_category = "error"
                    logger.error(f"No category scores generated for {image_path.name}")
                else:
                    best_category = max(category_scores, key=category_scores.get)
                    if confidence < self.image_confidence_threshold:
                        logger.debug(f"Low image confidence ({confidence:.3f} < {self.image_confidence_threshold:.3f}) for {image_path.name}, defaulting to neutral_content")
                        best_category = "neutral_content"

                duration = time.time() - analyze_start_time
                logger.info(f"Item {image_path.name} classified via IMAGE as: {best_category} in {duration:.2f}s. Confidence: {confidence:.3f}. (Scores: { {k: f'{v:.3f}' for k, v in category_scores.items()} })")
                return (best_category, "image")

        except UnidentifiedImageError:
            logger.error(f"Cannot identify image file (corrupted or wrong format): {image_path}")
            return ("error", "error")
        except FileNotFoundError:
             logger.error(f"Image file not found at path: {image_path}")
             return ("error", "error")
        except Exception as e:
            logger.exception(f"Unexpected error during image analysis for {image_path}: {e}")
            return ("error", "error")

    @modal.method()
    def process_item(self, item_dir_name: str):
        """
        Processes a single item directory: finds image, classifies, uploads all files.
        Designed to be called via .map().

        Args:
            item_dir_name: The name of the subdirectory within CONTAINER_DOWNLOADS_DIR to process.

        Returns:
            A dictionary summarizing the processing result for this item.
        """
        if not self.drive_service:
             msg = f"Skipping item '{item_dir_name}' due to missing Google Drive service in container."
             logger.error(msg)
             return {"item": item_dir_name, "status": "error", "reason": msg}
        if not self.category_folders:
             msg = f"Skipping item '{item_dir_name}' due to missing category folder configuration."
             logger.error(msg)
             return {"item": item_dir_name, "status": "error", "reason": msg}


        item_start_time = time.time()
        item_path = CONTAINER_DOWNLOADS_DIR / item_dir_name
        logger.info(f"Processing item directory: {item_path}")

        image_path = None
        files_to_upload = []
        category = "error" # Default category if issues arise early
        classification_method = "error" # Default if classification never runs
        target_category_folder_id = self.category_folders.get("error") # Default GDrive target

        if not item_path.is_dir():
            msg = f"Item path is not a directory: {item_path}. Skipping."
            logger.warning(msg)
            return {"item": item_dir_name, "status": "skipped", "reason": msg}

        # Find the first image file and collect all files for upload
        try:
            for file in item_path.iterdir():
                if file.is_file():
                    files_to_upload.append(file)
                    # Find the first image based on common extensions (case-insensitive)
                    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] and image_path is None:
                        image_path = file
                        logger.debug(f"Found image file: {file.name}")
            if not files_to_upload:
                logger.warning(f"No files found in directory: {item_path}")
                # Decide if this is an error or just skippable
                return {"item": item_dir_name, "status": "skipped", "reason": "No files in directory"}

        except Exception as e:
            logger.exception(f"Error scanning files in {item_path}: {e}")
            # Treat as error, attempt upload to error folder
            category = "error"
            target_category_folder_id = self.category_folders.get("error")
            return
        else:
            if not image_path:
                logger.warning(f"No image file found in {item_path}. Classifying item as 'error'.")
                category = "error"
                classification_method = "error"
            else:
                # Analyze the found image
                category, classification_method = self._analyze_image(image_path) # Returns ('error', 'error') on failure

            # Get the Google Drive folder ID for the determined category
            target_category_folder_id = self.category_folders.get(category)
            if not target_category_folder_id:
                logger.error(f"CRITICAL: No GDrive folder ID configured for category '{category}'. Uploading to error folder ID {self.category_folders.get('error')} instead.")
                target_category_folder_id = self.category_folders.get("error") # Fallback to error folder
                # If even the error folder ID is missing (checked in __enter__), we have a bigger problem
                if not target_category_folder_id:
                    msg = f"Cannot upload '{item_dir_name}', target category '{category}' AND error folder IDs are missing."
                    logger.critical(msg)
                    return {"item": item_dir_name, "status": "error", "reason": msg}
                
      

        # Create the specific subfolder for this item within the category folder
        # Use item_dir_name(shortcode) as the subfolder name in Google Drive
        new_folder , destination_subfolder_id = create_drive_folder(self.drive_service, item_dir_name, target_category_folder_id)
      
        if not new_folder:
            existing_folder_id = destination_subfolder_id
            logger.info(f"Item folder '{item_dir_name}' already exists in category '{category}' (ID: {existing_folder_id}). Skipping upload.")
            item_duration = time.time() - item_start_time # Include classification time if done above
            return {
                "item": item_dir_name,
                "status": "skipped_exist", # New status
                "category": category,
                "reason": "Subfolder already exists in Google Drive",
                "duration_seconds": round(item_duration, 2)
             }
   
        #fatal error 
        if not destination_subfolder_id:
            logger.error(f"Failed to create or find destination subfolder '{item_dir_name}' in category folder ID {target_category_folder_id}. Attempting upload to category folder root.")
            return 
            

        # Upload all collected files to the determined destination folder
        upload_count = 0
        upload_errors = 0
        if files_to_upload:
             logger.info(f"Uploading {len(files_to_upload)} files for '{item_dir_name}' to Drive folder ID {destination_subfolder_id} (Category: {category})")
             for file_path in files_to_upload:
                 file_id = upload_to_drive(self.drive_service, destination_subfolder_id, file_path)
                 if file_id:
                      upload_count += 1
                 else:
                      upload_errors += 1
                      logger.warning(f"Failed to upload file: {file_path.name}")

             logger.info(f"Finished uploading for '{item_dir_name}'. Success: {upload_count}, Errors: {upload_errors}")
    
        item_duration = time.time() - item_start_time
        final_status = "processed" if upload_errors == 0 else "processed_with_errors"
        if upload_count == 0 and upload_errors > 0:
             final_status = "error" # Treat as error if nothing could be uploaded


        return {
            "item": item_dir_name,
            "status": final_status,
            "category": category,
            "classification_method": classification_method,
            "files_found": len(files_to_upload),
            "uploads_successful": upload_count,
            "upload_errors": upload_errors,
            "duration_seconds": round(item_duration, 2)
        }
#(Run Once locally)
# Ensures the main GDrive folder exists before starting parallel processing.

@app.function(secrets=[modal.Secret.from_name(SECRET_NAME)])
def setup_drive_folders(req_parent_folder_id: str = None):
    """
    Creates the main classification folder in Google Drive if it doesn't exist.
    Returns the ID of this main classification folder.

    Args:
        req_parent_folder_id: Explicit GDrive folder ID to create the root folder under.
                              If None, uses GDRIVE_PARENT_FOLDER_ID env var or defaults to 'My Drive'.
    """

    setup_logger = logging.getLogger("setup_drive")
    setup_logger.setLevel(log_level) # Use global log level

    import json
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    parent_folder_id_to_use = req_parent_folder_id if req_parent_folder_id else GDRIVE_PARENT_FOLDER_ID
    setup_logger.info(f"Running setup. Target root folder name: '{ROOT_FOLDER_NAME}'. Parent ID: {parent_folder_id_to_use or 'My Drive root'}")


    try:
        with open(os.environ.get("SERVICE_ACCOUNT_JSON_PATH")) as f:
            service_account_info = json.load(f)
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        service = build("drive", "v3", credentials=creds)
        setup_logger.info("Google Drive service client built for setup.")
    except KeyError:
         setup_logger.error(f"SECRET ERROR: 'SERVICE_ACCOUNT_JSON' not found in environment for setup function. Ensure Modal secret '{SECRET_NAME}' is correct.")
         raise
    except Exception as e:
         setup_logger.exception("Failed to build Google Drive service client during setup.")
         raise


    storage_folder_id = create_drive_folder(service, ROOT_FOLDER_NAME, parent_folder_id_to_use)[1]

    if not storage_folder_id:
        setup_logger.critical(f"Failed to create or find the main storage folder '{ROOT_FOLDER_NAME}' under parent '{parent_folder_id_to_use or 'root'}'. Cannot proceed.")
        raise RuntimeError(f"Could not establish root Google Drive folder: {ROOT_FOLDER_NAME}")

    setup_logger.info(f"Setup complete. Main classification folder ('{ROOT_FOLDER_NAME}') ID: {storage_folder_id}")
    
    parent_classification_folder_id = storage_folder_id


    CATEGORY_DRIVE_NAMES = {
        "opioid_related": "Opioid Related",
        "neutral_content": "Neutral Content",
        "error": "Processing Errors" # Folder for items that fail processing
    }

    category_folders = {}
    logger.info(f"Ensuring category folders exist under parent folder ID: {parent_classification_folder_id}")
    for category_key, drive_folder_name in CATEGORY_DRIVE_NAMES.items():
        new_folder, folder_id = create_drive_folder(service, drive_folder_name, parent_classification_folder_id)
        if folder_id:
            category_folders[category_key] = folder_id
            # Write to .env file and flush immediately to ensure it's written before containers start
            with open(".env", "a") as f:
                f.write(f"\n{category_key.upper()}_FOLDER_ID={folder_id}\n")
                f.flush()  # Force write to disk immediately
                os.fsync(f.fileno())  # Ensure OS-level write completion
                
            logger.debug(f"Obtained folder ID for '{category_key}': {folder_id}")            
        else:
            logger.error(f"Failed to get or create Drive folder for category: {category_key} (Drive name: {drive_folder_name})")
            if category_key == "error":
                return None # Cannot proceed without an error folder

    logger.info(f"Category folder IDs obtained: {category_folders}")

    
    return category_folders

# Setup Drive folders at module level to ensure .env is written before containers start
# Only run setup if running locally (not in Modal container) and folder IDs are not already set
if modal.is_local() and GDRIVE_PARENT_FOLDER_ID:
    try:
        from dotenv import load_dotenv
        # Check if folder IDs are already set
        if not os.environ.get("OPIOID_RELATED_FOLDER_ID") or not os.environ.get("NEUTRAL_CONTENT_FOLDER_ID"):
            logger.info("Setting up Google Drive folder structure at module level...")
            setup_drive_folders.local(req_parent_folder_id=GDRIVE_PARENT_FOLDER_ID)
            # Reload .env after setup
            load_dotenv(override=True)
            logger.info("Drive folders setup complete, environment variables reloaded.")
    except Exception as e:
        logger.warning(f"Failed to setup Drive folders at module level: {e}. Will attempt in main() function.")

# --- Main Application Entrypoint ---

@app.local_entrypoint()
def main(drive_parent_id: str = None): # Allow overriding parent ID via CLI flag e.g. --drive-parent-id "..." useful for readme and deploying
    """
    Main entry point: Sets up Drive, lists items, runs parallel classification.
    """
    run_start_time = time.time()
    # Use the explicitly passed CLI flag highest precedence, then .env var, then None
    parent_id_for_setup = drive_parent_id if drive_parent_id else GDRIVE_PARENT_FOLDER_ID

    logger.info("--- Starting Classification Job ---")
    logger.info(f"Using local downloads source: {LOCAL_DOWNLOADS_DIR}")
    logger.info(f"Target container directory: {CONTAINER_DOWNLOADS_DIR}")
    logger.info(f"Modal App Name: {app.name}")
    logger.info(f"Requested GPU: {GPU_CONFIG}")

    # 1. Verify Drive folder setup (setup should have happened at module level)
    logger.info("Step 1: Verifying Google Drive folder structure...")
    if not os.environ.get("OPIOID_RELATED_FOLDER_ID") or not os.environ.get("NEUTRAL_CONTENT_FOLDER_ID"):
        logger.warning("Folder IDs not found, attempting setup now...")
        try:
            setup_drive_folders.local(req_parent_folder_id=parent_id_for_setup)
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except Exception as e:
            logger.exception("Failed to setup Google Drive structure. Exiting.")
            return


    # 2. List items (directories) to process from the container's perspective
    logger.info(f"Step 2: Scanning for item directories in container path: {CONTAINER_DOWNLOADS_DIR}")
    items_to_process = []
    try:
        # Check if the directory exists locally first (helps catch config errors early)
        if not LOCAL_DOWNLOADS_DIR.exists():
             logger.error(f"Local directory does not exist: {LOCAL_DOWNLOADS_DIR}. Check LOCAL_DOWNLOADS_DIR in config/env.")
             return
        if not LOCAL_DOWNLOADS_DIR.is_dir():
             logger.error(f"Local path is not a directory: {LOCAL_DOWNLOADS_DIR}.")
             return

        # since the  paths inside the container will mirror this structure under CONTAINER_DOWNLOADS_DIR.
        items_to_process = [d.name for d in LOCAL_DOWNLOADS_DIR.iterdir() if d.is_dir()]
        if not items_to_process:
            logger.warning(f"No subdirectories found in {LOCAL_DOWNLOADS_DIR}. Nothing to process.")
            return
        logger.info(f"Found {len(items_to_process)} potential item directories to process.")

    except Exception as e:
        logger.exception(f"Error listing local item directories in {LOCAL_DOWNLOADS_DIR}: {e}")
        return # Cannot proceed if listing fails

    # 3. Instantiate the Classifier class and process items in parallel using .map()
    logger.info(f"Step 3: Starting parallel processing for {len(items_to_process)} items...")
    classifier = Classifier() # Pass the folder ID to the container


    
    results = []
    map_start_time = time.time()

    # Use .map for parallel execution. `return_exceptions=True` allows processing to continue if one item fails.
    for result in classifier.process_item.map(items_to_process, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(result, Exception):
            # Log the exception traceback from the remote container
            logger.error(f"An exception occurred in remote processing: {result}", exc_info=False) # exc_info=False as traceback is in result
            results.append({"status": "framework_error", "error": str(result)})
        elif isinstance(result, dict):
            # Log summary from the returned dictionary
            logger.info(f"Processed '{result.get('item', 'N/A')}': Status={result.get('status', 'N/A')}, Category={result.get('category', 'N/A')}, Duration={result.get('duration_seconds', 'N/A')}s")
            results.append(result)
        else:
            # Handle unexpected return types
            logger.warning(f"Received unexpected result type from process_item.map: {type(result)}")
            results.append({"status": "unknown_result", "data": str(result)})


    map_duration = time.time() - map_start_time
    logger.info(f"Step 3 Complete. Parallel processing finished in {map_duration:.2f} seconds.")

    # 5. Summarize Results
    logger.info("--- Job Summary ---")
    processed_ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "processed")
    processed_w_errors = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "processed_with_errors")
    errors = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "error")
    framework_errors = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "framework_error")
    skipped = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
    skipped_exist = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped_exist")
    unknown = len(results) - (processed_ok + processed_w_errors + errors + framework_errors + skipped + skipped_exist)

    # Classification method breakdown
    classified_by_text = sum(1 for r in results if isinstance(r, dict) and r.get("classification_method") == "text")
    classified_by_image = sum(1 for r in results if isinstance(r, dict) and r.get("classification_method") == "image")
    classified_error = sum(1 for r in results if isinstance(r, dict) and r.get("classification_method") == "error")

    logger.info(f"Total items processed: {len(results)} / {len(items_to_process)}")
    logger.info(f"  Processed successfully: {processed_ok}")
    logger.info(f"  Processed with upload errors: {processed_w_errors}")
    logger.info(f"  Skipped (already exist in Drive): {skipped_exist}")
    logger.info(f"  Skipped (e.g., not dir, no files): {skipped}")
    logger.info(f"  Processing errors (classification/scan): {errors}")
    logger.info(f"  Framework/Container errors: {framework_errors}")
    if unknown > 0: logger.warning(f"  Unknown result status: {unknown}")
    logger.info(f"--- Classification Method Breakdown ---")
    logger.info(f"  Classified by text:  {classified_by_text}")
    logger.info(f"  Fell back to image:  {classified_by_image}")
    logger.info(f"  Classification error: {classified_error}")

    total_duration = time.time() - run_start_time
    logger.info(f"Total job duration: {total_duration:.2f} seconds.")
    logger.info("--- Classification Job Finished ---")