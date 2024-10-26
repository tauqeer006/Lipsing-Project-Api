import os
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from src.gradio_demo import SadTalker

app = FastAPI()

# Initialize the SadTalker model
checkpoint_path = 'checkpoints'
config_path = 'src/config'
sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)



@app.post("/generate-lipsync-video/")
async def generate_lip_sync_video(
    
    preprocess_type: str = Form('crop'),
    is_still_mode: bool = Form(False),
    enhancer: bool = Form(False),
    batch_size: int = Form(2),
    size_of_image: int = Form(256),
    pose_style: int = Form(0),
):
   
    print("Problem 1") 
    image_path = "C:/Users/tauqe/Desktop/New folder/sadtalker/SadTalker/b.jpg"  # Specify the image path here
    audio_path = "C:/Users/tauqe/Desktop/New folder/sadtalker/SadTalker/a.wav"  # Specify the audio path here
    output_path = './results/'

    os.makedirs("temp", exist_ok=True)

    print("Problem 2")
    try:
        result_path = sad_talker.test(
            source_image=image_path,
            driven_audio=audio_path,
            preprocess=preprocess_type,
            still_mode=is_still_mode,
            use_enhancer=enhancer,
            batch_size=batch_size,
            size=size_of_image,
            pose_style=pose_style,
            result_dir=os.path.dirname(output_path)) 
    except Exception as e:
        return {"error": str(e)}
    print("Problem 3")
    

    return FileResponse(result_path, media_type="video/mp4", filename="generated_video.mp4")
