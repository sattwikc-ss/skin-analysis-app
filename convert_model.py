from tensorflow.keras.models import load_model

# Load your old h5 model
model = load_model("skin_disease_model.h5", compile=False)

# Save it in new keras format
model.save("skin_disease_model.keras")
print("✅ Model converted and saved as skin_disease_model.keras")
