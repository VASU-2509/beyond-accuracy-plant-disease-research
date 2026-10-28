"""Conservative semantic mapping. Unlisted names are excluded, never guessed."""
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}
# Broad symptom names do not identify a compatible disease or pest reliably.
EXCLUSIONS = {"Corn leaf blight": "Blight subtype not specified",
              "Tomato mold leaf": "Mold etiology not specified",
              "Tomato two spotted spider mites leaf": "Pest identity/cropped-label spelling needs manual review"}
# These symptom-only labels are excluded even though a plausible mapping exists.
for _label in ["Bell_pepper leaf spot", "Tomato leaf yellow virus"]:
    PLANTDOC_TO_PLANTVILLAGE.pop(_label)
    EXCLUSIONS[_label] = "Symptom label insufficient to establish pathogen equivalence"

def map_label(label, classes):
    mapped = PLANTDOC_TO_PLANTVILLAGE.get(label)
    return mapped if mapped in classes else None
