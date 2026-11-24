import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import run_pipeline
from src.script_generator import ScenePrompt, ShortScript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("make_sun_short")

def main() -> int:
    scenes = [
        ScenePrompt(
            scene_id=0,
            narration="अगर कल सुबह अचानक सूरज गायब हो जाए... तो क्या होगा? क्या पृथ्वी तुरंत अंधेरे में डूब जाएगी?",
            display_text="Agar kal subah achanak Suraj gayab ho jaye… toh kya Earth turant andhere mein doob jayegi?",
            visual_prompt="Cinematic vertical 9:16 shot of the bright glowing Sun suddenly flickering and vanishing into total black void, leaving cosmic darkness in deep space, photorealistic 8k, Unreal Engine 5 space render",
            search_query="sun disappearing space",
            estimated_duration=6.0,
        ),
        ScenePrompt(
            scene_id=1,
            narration="सबसे शॉकिंग बात यह है... हमें अगले 8 मिनट 20 सेकंड्स तक पता भी नहीं चलेगा!",
            display_text="Sabse shocking baat… agle 8 minutes 20 seconds tak pata bhi nahi chalega!",
            visual_prompt="Cinematic vertical view of planet Earth peacefully glowing in sunlight with a glowing holographic digital countdown timer showing 08:20, photorealistic 8k space",
            search_query="earth sunlight orbit",
            estimated_duration=5.5,
        ),
        ScenePrompt(
            scene_id=2,
            narration="क्योंकि सूरज की रोशनी को पृथ्वी तक पहुंचने में... इतना ही वक्त लगता है।",
            display_text="Kyunki Suraj ki roshni ko Earth tak pahunchne mein itna hi time lagta hai.",
            visual_prompt="Photorealistic golden rays of sunlight traveling at light speed across deep dark space towards Earth, volumetric light beams, cosmic dust, cinematic 8k",
            search_query="light rays space earth",
            estimated_duration=5.0,
        ),
        ScenePrompt(
            scene_id=3,
            narration="उसके बाद... पूरी दुनिया में सन्नाटा और अंधेरा छा जाएगा। चांद भी गायब हो जाएगा, क्योंकि वो सूरज से चमकता है!",
            display_text="Uske baad poori duniya andhere mein doob jayegi. Chand bhi gayab lagega kyunki wo Suraj se chamakta hai!",
            visual_prompt="Planet Earth in complete terrifying darkness, night side with faint street lights, completely dark invisible Moon in background, deep stars, cinematic photorealistic",
            search_query="dark earth moon space",
            estimated_duration=7.5,
        ),
        ScenePrompt(
            scene_id=4,
            narration="और सबसे बड़ा ट्विस्ट... पृथ्वी अपनी कक्षा छोड़ देगी, और अंतरिक्ष में सीधी तैरने लगेगी!",
            display_text="Aur shocking fact… Earth apni orbit chhod kar space mein seedhi disha mein badhne lagegi!",
            visual_prompt="Planet Earth flying off its curved solar orbit trajectory in a straight vector line into the deep dark unknown galaxy void, cosmic physics simulation, cinematic 8k",
            search_query="earth flying space orbit",
            estimated_duration=7.0,
        ),
        ScenePrompt(
            scene_id=5,
            narration="कुछ ही दिनों में तापमान भयानक रूप से गिरेगा, समुद्र जम जाएंगे, और इंसानी जिंदगी खतरे में आ जाएगी!",
            display_text="Dheere-dheere temperature girne lagega, aur life ke liye conditions extremely dangerous ho jayengi.",
            visual_prompt="Planet Earth freezing into a giant snowball, frozen oceans covered in cracked blue ice sheets, snowy frozen skyscrapers in pitch darkness, cinematic photorealistic",
            search_query="frozen earth ice darkness",
            estimated_duration=7.0,
        ),
        ScenePrompt(
            scene_id=6,
            narration="सोचिए... हमारा पूरा वजूद सिर्फ एक तारे पर टिका है! क्या इंसान बिना सूरज के बच पाएंगे? कमेंट्स में बताइए!",
            display_text="Socho… poori duniya ek hi star par depend karti hai! Kya insaan bina Suraj ke bach payenge? Comments mein batao!",
            visual_prompt="Cinematic vertical view of a lone human in thermal space suit on a frozen cliff looking up at a sky filled with countless brilliant stars and distant nebulae, epic scale, 8k",
            search_query="astronaut on frozen earth",
            estimated_duration=7.0,
        ),
    ]

    custom_script = ShortScript(
        topic="What If The Sun Suddenly Disappeared?",
        title="Suraj Achanak Gayab Ho Jaye Toh? 🌍 #Shorts #Space",
        description="What happens if the Sun suddenly disappears from our solar system? Shocking space facts! #Shorts #Space #WhatIf #Earth #Science",
        tags=["space", "sun", "whatif", "earth", "science", "facts", "shorts"],
        scenes=scenes,
    )

    return run_pipeline(custom_script=custom_script)

if __name__ == "__main__":
    sys.exit(main())
