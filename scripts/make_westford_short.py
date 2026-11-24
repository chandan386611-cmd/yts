import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import run_pipeline
from src.script_generator import ScenePrompt, ShortScript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("make_westford_short")

def main() -> int:
    scenes = [
        ScenePrompt(
            scene_id=0,
            narration='Kya aap jaante hain? <break time="300ms"/> America ne ek baar Earth ke chaaron taraf artificial ring banane ki koshish ki thi!',
            display_text="Kya aap jaante hain? America ne ek baar Earth ke around artificial ring banane ki koshish ki thi!",
            visual_prompt="Cinematic vertical 9:16 shot of planet Earth from deep space encircled by a shimmering metallic ring of copper needles glittering under cosmic sunlight, photorealistic 8k, Unreal Engine 5 space render",
            search_query="earth metallic artificial ring space",
            estimated_duration=6.0,
        ),
        ScenePrompt(
            scene_id=1,
            narration="1960s Cold War mein America ko dar tha ki nuclear war ke dauran unka radio communication destroy ho sakta hai.",
            display_text="1960s Cold War mein America ko dar tha ki nuclear war mein radio communication destroy ho sakti hai.",
            visual_prompt="Tense vintage 1960s military war room bunker with radar screens, analog telecommunications, and defense generals staring at world maps, cinematic moody lighting 8k",
            search_query="cold war military radar bunker 1960s",
            estimated_duration=6.5,
        ),
        ScenePrompt(
            scene_id=2,
            narration="Isi dar se US scientists ne design kiya Project West Ford, jo sunne mein poora science-fiction lagta hai!",
            display_text="Isi dar se scientists ne design kiya Project West Ford, jo sunne mein science-fiction lagta hai!",
            visual_prompt="Secret 1960s aerospace research laboratory with engineers examining blueprints showing Earth surrounded by a metallic dipole ring, dramatic vintage cinematic lighting 8k",
            search_query="aerospace blueprints project westford",
            estimated_duration=6.0,
        ),
        ScenePrompt(
            scene_id=3,
            narration="Unhone space mein ek rocket se 480 million microscopic copper needles release kar diye!",
            display_text="Unhone space mein 480 million microscopic copper dipoles release kar diye!",
            visual_prompt="Space satellite dispenser opening in high Earth orbit releasing hundreds of millions of microscopic shiny copper needles forming a shimmering cloud, 8k cinematic space render",
            search_query="satellite releasing copper needles space",
            estimated_duration=5.5,
        ),
        ScenePrompt(
            scene_id=4,
            narration="Plan tha ki ye tiny copper needles radio signals reflect karne wali ek artificial belt bana sakein.",
            display_text="Plan tha ki ye tiny copper needles radio signals reflect karne wali artificial belt banayein.",
            visual_prompt="Radio frequency transmission beams beaming up from a giant Earth satellite dish bouncing off a metallic reflective cloud in orbit back to Earth, cosmic illustration 8k",
            search_query="radio signals bouncing space metallic ring",
            estimated_duration=6.0,
        ),
        ScenePrompt(
            scene_id=5,
            narration="Lekin duniya bhar ke astronomers ne virodh kiya, aur technical problems ki wajah se project band ho gaya.",
            display_text="Lekin astronomers ke virodh aur technical problems ki wajah se experiment band ho gaya.",
            visual_prompt="Silhouette of giant astronomical observatory telescope pointing at night sky with astronomers looking up at space debris glittering overhead, moody starlight 8k",
            search_query="astronomers observatory telescope space debris",
            estimated_duration=6.0,
        ),
        ScenePrompt(
            scene_id=6,
            narration='Socho, insaan ne Earth ke around ring banane ka plan banaya tha! <break time="300ms"/> Genius tha ya dangerous? Comments mein batao!',
            display_text="Insaan ne Earth ke around ring banane ka plan banaya tha! Genius tha ya dangerous? Comments mein batao!",
            visual_prompt="Breathtaking cinematic wide shot of planet Earth with a faint shimmering copper ring around it, deep space stars and blue atmospheric glow, epic scale 8k",
            search_query="earth with shimmering ring deep space",
            estimated_duration=6.0,
        ),
    ]

    custom_script = ShortScript(
        topic="Project West Ford: When America Tried to Build an Artificial Ring Around Earth",
        title="America Ne Banayi Earth Ki Artificial Ring? [Project West Ford] #Shorts #SpaceFacts",
        description="In the 1960s Cold War, the US launched 480 million copper needles into space to create an artificial ring around Earth! Discover Project West Ford. #Shorts #SpaceFacts #ProjectWestFord #ColdWar #Science #Zorvian",
        tags=["project west ford", "space", "science", "earth ring", "cold war", "space facts", "shorts", "hindi facts"],
        scenes=scenes,
    )

    return run_pipeline(custom_script=custom_script)

if __name__ == "__main__":
    sys.exit(main())
