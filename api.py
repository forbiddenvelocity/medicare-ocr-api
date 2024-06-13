from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import NougatProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
import logging
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

nltk.download('punkt')

# Load the processor and model
processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Prepare the image for the model
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Generate the transcription
        outputs = model.generate(
            pixel_values.to(device),
            min_length=1,
            max_new_tokens=300,  # Adjust as needed
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )

        # Decode and post-process the generated sequence
        sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        sequence = processor.post_process_generation(sequence, fix_markdown=False)

        # Split the text into lines and return as JSON
        lines = sequence.split('\n')
        return JSONResponse(content={"lines": lines})
    except Exception as e:
        logger.error(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
