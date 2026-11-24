import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import run_pipeline
from src.script_generator import ScenePrompt, ShortScript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("make_goa_short")

def main() -> int:
    scenes = [
        ScenePrompt(
            scene_id=0,
            narration='Kya tum jaante ho? <break time="300ms"/> India ne sirf 36 ghanton mein, Goa se 450 saal purana Portuguese rule ukhaad phenka tha!',
            display_text="Kya tum jaante ho? India ne sirf 36 ghanton mein 450 saal purana Portuguese rule khatam kiya tha!",
            visual_prompt="Cinematic vertical 9:16 shot of Indian national flag tricolor unfurling heroically over historic Aguada stone fortress in Goa, golden sunrise light, 1961 military victory, 8k photorealistic",
            search_query="indian flag goa fort",
            estimated_duration=5.5,
        ),
        ScenePrompt(
            scene_id=1,
            narration="Portugal ne 1510 mein Goa par kabza kiya tha. Aur jab 1947 mein Bharat azaad hua, tab bhi Goa azaad nahi tha!",
            display_text="Portugal ne 1510 mein kabza kiya tha. 1947 mein azaadi ke baad bhi Goa gulami mein tha!",
            visual_prompt="Vintage 16th century wooden Portuguese naval galleon warship anchored off the tropical rocky coast of Goa in 1510, dramatic dark ocean waves, cinematic historical painting style 8k",
            search_query="portuguese warship goa 1510",
            estimated_duration=6.5,
        ),
        ScenePrompt(
            scene_id=2,
            narration="Bharat ne 14 saal tak shanti se baat ki, lekin Portuguese dictator ne Goa chhodne se saaf inkaar kar diya.",
            display_text="Bharat ne 14 saal shanti se demand ki, lekin Portuguese dictator ne Goa chhodne se inkaar kiya!",
            visual_prompt="Tense historical 1950s diplomatic negotiation room with vintage map of India showing Goa territory highlighted in red, vintage microphones and serious diplomats, dramatic cinematic lighting 8k",
            search_query="diplomats map goa negotiation",
            estimated_duration=6.0,
        ),
        ScenePrompt(
            scene_id=3,
            narration="Phir 18 December 1961 ko, India ne launch kiya Operation Vijay! Army, Navy aur Air Force ne teeno taraf se Goa ko gher liya!",
            display_text="Phir 18 December 1961 ko launch hua Operation Vijay! Army, Navy aur Air Force ne Goa ko gher liya!",
            visual_prompt="Indian military army tanks and convoy advancing across Goa border with fighter jets overhead and warships off the coast, Operation Vijay 1961, dynamic dust, photorealistic 8k",
            search_query="indian army tanks operation vijay",
            estimated_duration=7.0,
        ),
        ScenePrompt(
            scene_id=4,
            narration="Hamari fauj ke aage Portuguese resistance dher ho gaya! Sirf 36 ghanton mein unke Governor-General ne surrender kar diya.",
            display_text="Hamari fauj ke aage resistance dher ho gaya! 36 ghanton mein Governor-General ne surrender kiya!",
            visual_prompt="Portuguese military commander signing the official surrender instrument papers at an army table with Indian military officers standing tall, December 1961 Goa, cinematic historical photorealistic 8k",
            search_query="military surrender signing 1961",
            estimated_duration=6.5,
        ),
        ScenePrompt(
            scene_id=5,
            narration="Aur is tarah, Bharat ki dharti se 451 saal purana colonial rule hamesha ke liye dafan ho gaya!",
            display_text="Aur is tarah Bharat ki dharti se 451 saal purana colonial rule hamesha ke liye khatam hua!",
            visual_prompt="Triumphant Indian soldiers in open military jeeps riding through cheering crowds on the streets of Panaji Goa, civilians waving Indian flags, confetti in air, historic freedom celebration 8k",
            search_query="liberation panaji goa celebration",
            estimated_duration=6.0,
        ),
        ScenePrompt(
            scene_id=6,
            narration='Isi tareekh 19 December ko aaj hum Goa Liberation Day kehte hain! <break time="300ms"/> Proud Indian ho toh Jai Hind zaroor likhein!',
            display_text="19 December ko aaj hum Goa Liberation Day kehte hain! Proud Indian ho toh Jai Hind zaroor likhein!",
            visual_prompt="Majestic vertical aerial view of sunny Goa coastline with Aguada fort and a massive Indian national flag waving proudly against sunset sky, epic scale cinematic 8k",
            search_query="goa coast indian flag sunset",
            estimated_duration=6.0,
        ),
    ]

    custom_script = ShortScript(
        topic="India vs Portugal: The 36-Hour Operation (Goa Liberation 1961)",
        title="India vs Portugal: 36 Ghanton Mein Goa Azaad! 🇮🇳 #Shorts #History",
        description="How Indian Armed Forces liberated Goa from 451 years of Portuguese colonial rule in just 36 hours during Operation Vijay (1961)! #Shorts #Goa #IndianArmy #History #OperationVijay #JaiHind",
        tags=["goa", "history", "indianarmy", "operationvijay", "portugal", "shorts", "india", "facts"],
        scenes=scenes,
    )

    return run_pipeline(custom_script=custom_script)

if __name__ == "__main__":
    sys.exit(main())
