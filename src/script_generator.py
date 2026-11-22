"""
Structured Script & Visual Prompt Generator using Google Gemini API.

Performs topic deduplication against past history, generates high-retention
structured JSON scripts formatted into 9-11 scenes with 105-120 total words,
applies art-direction locks, and handles duration-feedback re-prompting.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        def decorator(f: Any) -> Any:
            return f
        return decorator

    def retry_if_exception_type(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return None

    def stop_after_attempt(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return None

    def wait_exponential(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return None

from src.config import config

import requests

logger = logging.getLogger("script_generator")


class ScenePrompt(BaseModel):
    """Individual scene script and visual direction."""
    scene_id: int
    narration: str
    display_text: str | None = None
    visual_prompt: str
    search_query: str | None = None
    estimated_duration: float = Field(default=4.0, description="Estimated duration in seconds")

    @property
    def full_visual_prompt(self) -> str:
        """Visual prompt concatenated with mandatory art-direction style suffix."""
        cleaned = self.visual_prompt.strip().rstrip(",")
        return f"{cleaned}, {config.ART_STYLE_SUFFIX}"

    @property
    def image_search_query(self) -> str:
        """2-4 clean keywords for retrieving relevant real-world photographs."""
        if self.search_query and self.search_query.strip():
            return self.search_query.strip()
        # Fallback: extract main non-stopword tokens from visual prompt
        tokens = re.findall(r"[A-Za-z0-9]+", self.visual_prompt)
        stopwords = {
            "a", "an", "the", "in", "on", "at", "by", "with", "of", "and", "or", "for",
            "from", "to", "showing", "scene", "view", "extreme", "macro", "close", "up",
            "wide", "shot", "high", "angle", "dramatic", "cinematic", "lighting", "under"
        }
        filtered = [w for w in tokens if w.lower() not in stopwords]
        return " ".join(filtered[:3]) if filtered else "ancient mystery"


class ShortScript(BaseModel):
    """Complete structured YouTube Short script."""
    topic: str
    title: str
    description: str
    tags: list[str]
    scenes: list[ScenePrompt]

    @property
    def total_words(self) -> int:
        """Calculate total spoken words across all scene narrations."""
        return sum(len(scene.narration.split()) for scene in self.scenes)

    @property
    def full_narration_text(self) -> str:
        """Merged narration text."""
        return " ".join(scene.narration.strip() for scene in self.scenes)


def load_recent_topics(max_history: int = 40) -> list[str]:
    """Load up to max_history recently used topics from history.json."""
    if not config.HISTORY_FILE.exists():
        return []
    try:
        with open(config.HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        used = data.get("used_topics", [])
        # Return last 40 entries
        return [entry if isinstance(entry, str) else entry.get("topic", "") for entry in used][-max_history:]
    except Exception as e:
        logger.warning("Could not read recent topics: %s", e)
        return []


def _build_system_prompt(recent_topics: list[str], niche: str) -> str:
    """Build the strict system instructions for Gemini."""
    avoid_list = "\n".join(f"- {t}" for t in recent_topics if t)
    is_hinglish = getattr(config, "LANGUAGE", "hinglish").lower() in ("hinglish", "hindi")

    if is_hinglish:
        language_rule = (
            "LANGUAGE & AUDIO DYNAMICS REQUIREMENT (CRITICAL FOR VIRAL RETENTION):\n"
            "1. 'narration': Spoken voiceover text. MUST be written in natural, cinematic conversational HINDI (Devanagari script, e.g. 'इमेजिन कीजिए...', 'दरअस्ल एक खौफनाक सच यह है...'). "
            "You can include common English loan words in English letters (e.g., 'Brain', 'Secret', 'Dark Psychology', 'Mirroring Trap').\n"
            "   AUDIO PUNCTUATION RULES (MANDATORY FOR STUDIO VOCAL DELIVERY):\n"
            "   - Use '...' (ellipses) before shocking revelations, twists, and dramatic pauses (creates a natural 300ms suspense pause in neural audio).\n"
            "   - Use ',' (commas) after conversational openers and natural breathing points.\n"
            "   - Use '?' (question marks) on opening hooks and curiosity moments (forces upward vocal pitch inflection that grabs human attention).\n"
            "   - Use '—' (em-dashes) for sudden mid-sentence mental pivots.\n"
            "   - Use '!' selectively for impactful punchlines.\n"
            "   - Use '।' (purna viram) or '.' for clean, confident sentence conclusions.\n"
            "2. 'display_text': On-screen subtitle text. MUST be written in punchy Romanized HINGLISH (Hindi in English/Latin letters, e.g., 'Imagine kijiye...', 'Ek khaufnaak sach...').\n"
            "3. 'title' & 'description': Write in high-CTR, high-curiosity Hinglish with viral hashtags.\n"
            "4. 'search_query': MUST BE 2-3 CLEAN ENGLISH KEYWORDS matching real stock video footage (e.g., 'interrogation', 'shadow', 'human eye', 'people talking', 'dark room', 'chess', 'night walking').\n"
            "5. 'visual_prompt': MUST ALWAYS BE IN ENGLISH for the AI video generator."
        )
    else:
        language_rule = (
            "LANGUAGE: All narration, display_text, title, and description MUST be written in punchy, captivating English with expressive punctuation (... for suspense, ? for pitch inflection, — for dramatic pivots). "
            "Visual prompts must be in English."
        )

    niche_words = re.findall(r"[A-Za-z]+", niche)
    clean_tag = niche_words[0].capitalize() if niche_words else "Facts"

    return f"""You are an elite Hollywood documentary filmmaker and viral YouTube Shorts thriller writer in the niche: '{niche}'.
Your scripts do NOT sound like dry textbooks, Wikipedia articles, or academic lectures. Instead, you craft intense, addictive micro-thrillers inspired by Christopher Nolan and high-retention documentary mysteries.

CRITICAL 4-BEAT THRILLER STORYTELLING ARC:
1. Beat 1: The Shocking Cold Open (Scene 0)
   - Under 13 words. Shatter common sense or warn the viewer about an invisible reality or mind-blowing breakthrough in '{niche}'.
   - ZERO fluff, no 'Hello friends' or intro titles.
2. Beat 2: The Deepening Mystery & The Hidden Mechanism (Scenes 1 to 3)
   - Unfold the classified experiment, futuristic technology, cosmic anomaly, or hidden mechanism with sensory tension.
3. Beat 3: The Chilling Twist & Reality Shift (Scenes 4 to 6)
   - The mind-blowing realization: this changes everything we thought we knew about our world, brain, or future.
4. Beat 4: The Provocative Cliffhanger & Comment Trigger (Scenes 7 to N-1)
   - Give the viewer a thought-provoking perspective or question that forces them to comment and debate.

CRITICAL RETENTION & TIMING RULES:
1. {language_rule}
2. TOTAL SPOKEN WORDS: MUST BE STRICTLY BETWEEN {config.TARGET_MIN_WORDS} AND {config.TARGET_MAX_WORDS} WORDS TOTAL across the entire script. Spoken at an energetic brisk pace (150-165 WPM), this results in exactly 40.0 to 45.0 seconds of audio.
3. SCENE COUNT: EXACTLY {config.MIN_SCENES} TO {config.MAX_SCENES} DISTINCT SCENES (numbered 0 to N-1).
4. PACING: Each individual scene narration must be 9 to 14 words (~3.5 to 4.2 seconds).
5. STRICT DEDUPLICATION: DO NOT reuse or closely duplicate any of the following recently covered topics:
{avoid_list if avoid_list else "(No prior topics - pick an ultra-viral breakthrough fact)"}

OUTPUT FORMAT:
You MUST respond ONLY with a valid raw JSON object matching this exact structure:
{{
  "topic": "Concise topic title",
  "title": "Viral Hook Title Under 60 Chars #Shorts #{clean_tag}",
  "description": "Engaging 2-3 sentence description with high SEO keywords and hashtags #{clean_tag} #Shorts #Viral",
  "tags": ["{clean_tag.lower()}", "shorts", "viral", "facts"],
  "scenes": [
    {{
      "scene_id": 0,
      "narration": "इमेजिन कीजिए... कोई बात करते वक्त, चुपचाप आपका दिमाग स्कैन कर रहा है।",
      "display_text": "Imagine kijiye... koi baat karte waqt chupchap aapka dimag scan kar raha hai.",
      "visual_prompt": "Cinematic visual description in English for AI video generator",
      "search_query": "stock video keywords",
      "estimated_duration": 4.1
    }}
  ]
}}"""


def _parse_gemini_response(response_text: str) -> ShortScript:
    """Clean and parse JSON from Gemini model output."""
    cleaned = response_text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Fallback regex extraction of json object
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise ValueError(f"Failed to parse JSON from Gemini response: {exc}\nRaw: {response_text[:300]}") from exc

    scenes = [ScenePrompt(**s) for s in data["scenes"]]
    niche_words = re.findall(r"[A-Za-z]+", config.NICHE_TOPIC)
    default_tags = [w.lower() for w in niche_words[:3]] + ["shorts", "viral"]
    script = ShortScript(
        topic=data["topic"],
        title=data["title"],
        description=data["description"],
        tags=data.get("tags") or default_tags,
        scenes=scenes,
    )
    return script


def generate_fallback_offline_script(niche: str) -> ShortScript:
    """Generate a high-quality curated offline script matched to the requested niche if Gemini API is unavailable."""
    logger.info("Generating curated offline fallback script for niche: %s", niche)
    niche_lower = (niche or "").lower()
    is_hinglish = getattr(config, "LANGUAGE", "hinglish").lower() in ("hinglish", "hindi")

    # 1. Science, Future & Technology Niche
    if any(k in niche_lower for k in ["tech", "science", "future", "ai", "robot", "quantum", "cyber", "digital"]):
        if is_hinglish:
            scenes = [
                ScenePrompt(
                    scene_id=0,
                    narration="इमेजिन कीजिए... 2035 में आपका दिमाग सीधे इंटरनेट से कनेक्ट हो जाए।",
                    display_text="Imagine kijiye... 2035 mein aapka dimag internet se connect ho jaye.",
                    visual_prompt="Futuristic human silhouette with glowing cybernetic neural connections in head",
                    search_query="technology",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=1,
                    narration="साइंटिस्ट्स का दावा है कि न्यूरालिंक और क्वांटम कंप्यूटर्स इसे मुमकिन कर रहे हैं।",
                    display_text="Scientists ka dawa hai ki Neuralink aur Quantum computers ise mumkin kar rahe hain.",
                    visual_prompt="High tech laboratory with glowing quantum supercomputer core",
                    search_query="computer",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=2,
                    narration="अब आपको कुछ याद करने या गूगल पर टाइप करने की जरूरत नहीं पड़ेगी।",
                    display_text="Ab aapko kuch yaad karne ya type karne ki zaroorat nahi padegi.",
                    visual_prompt="Person wearing subtle futuristic cybernetic glasses looking at digital holograms",
                    search_query="virtual reality",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=3,
                    narration="सिर्फ एक सोच... और पूरी दुनिया का डेटा आपकी आंखों के सामने तैरने लगेगा।",
                    display_text="Sirf ek soch... aur poori duniya ka data aapki aankhon ke saamne hoga.",
                    visual_prompt="Futuristic digital interface data streams glowing in dark room",
                    search_query="digital data",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=4,
                    narration="लेकिन ठहरिए... इस सुपर-इंटेलिजेंस का एक खौफनाक डार्क साइड भी है।",
                    display_text="Lekin thehriye... is super intelligence ka ek khaufnaak dark side bhi hai.",
                    visual_prompt="Dramatic warning red digital glitches across glowing motherboard",
                    search_query="server room",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=5,
                    narration="अगर आपका दिमाग ऑनलाइन होगा, तो क्या कोई आपके विचारों को हैक कर सकेगा?",
                    display_text="Agar aapka dimag online hoga, toh kya koi aapke vichaaron ko hack kar sakega?",
                    visual_prompt="Shadowy hooded hacker sitting in front of neon computer screens",
                    search_query="hacker",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=6,
                    narration="आपकी पर्सनल यादें, बैंक पासवर्ड्स और सीक्रेट्स—एक क्लिक में बिक सकते हैं।",
                    display_text="Aapki personal yaadein aur secrets—ek click mein bik sakte hain.",
                    visual_prompt="Abstract digital lock breaking open into glowing code fragments",
                    search_query="cyber security",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=7,
                    narration="टेक कंपनियां दावा करती हैं कि बायोमेट्रिक फायरवॉल्स आपको हमेशा प्रोटेक्ट रखेंगी।",
                    display_text="Tech companies dawa karti hain ki firewalls aapko protect rakhengi.",
                    visual_prompt="Futuristic robotic arm assembling high precision microchips",
                    search_query="robot arm",
                    estimated_duration=3.8,
                ),
                ScenePrompt(
                    scene_id=8,
                    narration="लेकिन सवाल यह है... क्या मशीन कभी इंसानी सोच से ज्यादा ताकतवर बन जाएगी?",
                    display_text="Lekin sawal yeh hai... kya machine insaan se zyada taakatwar ban jayegi?",
                    visual_prompt="Humanoid robot with glowing blue optical eyes looking at human hand",
                    search_query="humanoid robot",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=9,
                    narration="क्या आप अपने दिमाग में न्यूरल चिप लगवाएंगे? अपनी राय कमेंट्स में बताइए।",
                    display_text="Kya aap apne dimag mein chip lagwayenge? Comments mein batayein!",
                    visual_prompt="Sprawling futuristic city skyline at night with flying vehicles and neon lights",
                    search_query="future city",
                    estimated_duration=4.3,
                ),
            ]
            return ShortScript(
                topic="Brain-Computer Interfaces & The 2035 Neural Singularity",
                title="2035 Mein Insaan Amar Ho Jayenge? #Shorts #Tech #Future",
                description="Kya 2035 tak humara dimaag cloud par upload ho jayega? Quantum computers aur nanobots ka shocking sach! #Science #Technology #Future #AI #Shorts",
                tags=["science", "technology", "future", "ai", "tech", "shorts", "quantum"],
                scenes=scenes,
            )
        else:
            scenes = [
                ScenePrompt(
                    scene_id=0,
                    narration="Imagine waking up in 2035 with your conscious mind plugged into the web.",
                    visual_prompt="Futuristic human silhouette with glowing cybernetic neural connections in head",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=1,
                    narration="Quantum computers and neural interfaces are rapidly bringing the singularity to human life.",
                    visual_prompt="High tech laboratory with glowing quantum supercomputer core",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=2,
                    narration="You will never need to memorize facts or touch a smartphone again.",
                    visual_prompt="Person wearing subtle futuristic cybernetic glasses looking at digital holograms",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=3,
                    narration="A single fleeting thought, and humanity's collective knowledge flashes before your eyes.",
                    visual_prompt="Futuristic digital interface data streams glowing in dark room",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=4,
                    narration="Yet this superhuman cognitive leap conceals an alarming, existential vulnerability.",
                    visual_prompt="Dramatic warning red digital glitches across glowing motherboard",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=5,
                    narration="If human thoughts become network packets, can dark actors hijack your perception?",
                    visual_prompt="Shadowy hooded hacker sitting in front of neon computer screens",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=6,
                    narration="Your most intimate private memories could be extracted and traded overnight.",
                    visual_prompt="Abstract digital lock breaking open into glowing code fragments",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=7,
                    narration="Technologists promise unbreakable biological firewalls will protect personal autonomy forever.",
                    visual_prompt="Futuristic robotic arm assembling high precision microchips",
                    estimated_duration=3.8,
                ),
                ScenePrompt(
                    scene_id=8,
                    narration="Will human wisdom stay in command, or will synthetic minds surpass us?",
                    visual_prompt="Humanoid robot with glowing blue optical eyes looking at human hand",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=9,
                    narration="Would you agree to merge your mind with AI? Tell us below.",
                    visual_prompt="Sprawling futuristic city skyline at night with flying vehicles and neon lights",
                    estimated_duration=4.3,
                ),
            ]
            return ShortScript(
                topic="Brain-Computer Interfaces & The 2035 Singularity",
                title="Will Humans Merge With AI By 2035? #Shorts #Tech",
                description="Neural interfaces and quantum leaps are bridging humanity and cyberspace. What does the future hold? #Technology #Science #Future #AI #Shorts",
                tags=["technology", "science", "future", "ai", "singularity", "shorts"],
                scenes=scenes,
            )

    # 2. Space, Cosmos & Astronomy Niche
    if any(k in niche_lower for k in ["space", "cosmos", "universe", "astronomy", "alien", "galaxy"]):
        if is_hinglish:
            scenes = [
                ScenePrompt(
                    scene_id=0,
                    narration="इमेजिन कीजिए... हमारे ब्रह्मांड में एक ऐसी जगह है, जहां सिर्फ खौफनाक सन्नाटा है।",
                    display_text="Imagine kijiye... brahmand mein ek aisi jagah hai jahan khaufnaak sannata hai.",
                    visual_prompt="Deep outer space with distant shimmering stars and cosmic nebula",
                    search_query="outer space",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=1,
                    narration="इसे Boötes Void कहते हैं—जो 330 मिलियन लाइट इयर्स में फैला विशाल शून्य है।",
                    display_text="Ise Boötes Void kehte hain—jo 330 million light-years lamba shunya hai.",
                    visual_prompt="Vast empty dark void in space surrounded by distant spiral galaxies",
                    search_query="galaxy",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=2,
                    narration="नॉर्मल स्पेस के हिसाब से, यहां कम से कम 10,000 गैलेक्सीज होनी चाहिए थीं।",
                    display_text="Normal space ke hisaab se yahan 10,000 galaxies honi chahiye thin.",
                    visual_prompt="Cluster of glowing spiral galaxies spinning slowly in deep space",
                    search_query="space station",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=3,
                    narration="लेकिन एस्ट्रोनॉमर्स को पूरे इलाके में... सिर्फ कुछ गिनी-चुनी गैलेक्सीज ही मिलीं।",
                    display_text="Lekin astronomers ko yahan sirf kuch gini-chuni galaxies hi milin.",
                    visual_prompt="Giant optical telescope observatory dome open toward starlit night sky",
                    search_query="telescope",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=4,
                    narration="बाकी पूरा इलाका बिल्कुल वीरान, ठंडा और भयानक रूप से खाली पड़ा है।",
                    display_text="Baki poora ilaka bilkul veeran aur bhayanak roop se khali pada hai.",
                    visual_prompt="Mysterious dark celestial sphere absorbing starlight in cold deep space",
                    search_query="black hole",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=5,
                    narration="वैज्ञानिक हैरान हैं... क्या किसी अज्ञात महाशक्ति ने इन अरबों तारों को निगल लिया?",
                    display_text="Scientists heran hain... kya kisi shakti ne in arbon taron ko nigal liya?",
                    visual_prompt="Supermassive black hole gravitational lensing warping surrounding light",
                    search_query="supernova",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=6,
                    narration="या फिर कोई एडवांस टाइप-3 सिविलाइजेशन सारे तारों की एनर्जी हार्वेस्ट कर रही है?",
                    display_text="Ya koi advanced civilization saare taron ki energy harvest kar rahi hai?",
                    visual_prompt="Gigantic Dyson sphere mega-structure encircling a glowing cosmic star",
                    search_query="sci fi space",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=7,
                    narration="अगर आप इस वॉइड के बीच हों, तो चारों तरफ सिर्फ अंतहीन अंधेरा दिखेगा।",
                    display_text="Agar aap is void ke beech hon, toh chaaron taraf sirf andhera dikhega.",
                    visual_prompt="Lone astronaut floating in the pitch-black silence of endless deep space",
                    search_query="astronaut",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=8,
                    narration="कोई भी तारा या रोशनी आपको कभी दिखाई नहीं देगी।",
                    display_text="Koi bhi tara ya roshni aapko kabhi dikhayi nahi degi.",
                    visual_prompt="Planet Earth isolated against infinite black void of space",
                    search_query="planet earth",
                    estimated_duration=3.7,
                ),
                ScenePrompt(
                    scene_id=9,
                    narration="क्या ब्रह्मांड में हम वाकई अकेले हैं? अपने विचार कमेंट्स में जरूर बताइए।",
                    display_text="Kya brahmand mein hum akele hain? Comments mein zaroor batayein.",
                    visual_prompt="Milky way galaxy arching across a silent starry midnight horizon",
                    search_query="night sky stars",
                    estimated_duration=4.3,
                ),
            ]
            return ShortScript(
                topic="The Boötes Void & The Great Cosmic Silence",
                title="Space Ka Sabse Khaufnaak Sannata! #Shorts #Space",
                description="Universe mein 330 million light years lamba khali andhera! Boötes Void ka unheard rahasya. #Space #Universe #Cosmos #Shorts #Mystery",
                tags=["space", "universe", "cosmos", "astronomy", "shorts", "mystery"],
                scenes=scenes,
            )

    # 3. History & Ancient Mysteries Niche
    if any(k in niche_lower for k in ["history", "ancient", "mystery", "archaeology", "ruin"]):
        if is_hinglish:
            scenes = [
                ScenePrompt(
                    scene_id=0,
                    narration="इमेजिन कीजिए... अपने घर का तहखाना खोदते वक्त, आपको एक गुप्त शहर मिल जाए।",
                    display_text="Imagine kijiye... ghar ke niche aapko ek gupt shehar mil jaye.",
                    visual_prompt="Dark antique stone cellar stairs leading down into a vast subterranean cavern",
                    search_query="ancient cave",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=1,
                    narration="1963 में तुर्की के एक आदमी ने दीवार तोड़ी, और वो दंग रह गया।",
                    display_text="1963 mein Turkey ke ek aadmi ne deewar todi aur woh dang reh gaya.",
                    visual_prompt="Archaeologist holding torch illuminating ancient underground stone tunnels",
                    search_query="ancient ruins",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=2,
                    narration="ज़मीन के 85 मीटर नीचे... 20,000 इंसानों के रहने की सीक्रेट जगह छिपी थी।",
                    display_text="Zameen ke 85 meter niche 20,000 logon ka gupt shehar chhipa tha.",
                    visual_prompt="Massive underground stone city carved directly into volcanic rock",
                    search_query="underground stone",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=3,
                    narration="इसे डेरिनकुयू कहते हैं—जिसमें 18 लेवल्स, स्कूल, मंदिर और ताजी हवा की शाफ्ट्स थीं।",
                    display_text="Ise Derinkuyu kehte hain—jisme 18 levels aur secret rooms the.",
                    visual_prompt="Ancient massive round stone millstone door blocking a subterranean entrance",
                    search_query="stone door",
                    estimated_duration=4.1,
                ),
                ScenePrompt(
                    scene_id=4,
                    narration="सबसे रहस्यमयी बात यह है कि... इसे हजारों साल पहले बिना मशीनों के कैसे तराशा गया?",
                    display_text="Sabse bada rahasya... ise bina machines ke kaise tarasha gaya?",
                    visual_prompt="Chiseled ancient carved stone pillars and archways deep underground",
                    search_query="ancient pillar",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=5,
                    narration="भारी पत्थर के दरवाजे ऐसे बने थे कि सिर्फ अंदर से बंद हो सकें।",
                    display_text="Bhari patthar ke darwaze aise bane the jo sirf andar se band hote the.",
                    visual_prompt="Silhouette of a torchbearer walking through claustrophobic ancient corridors",
                    search_query="torchlight",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=6,
                    narration="आखिर हमारे पूर्वज... ज़मीन के ऊपर किस भयानक खतरे से बचने के लिए छुपे थे?",
                    display_text="Aakhir purvaj zameen ke upar kis khatre se bachne ke liye chhupe the?",
                    visual_prompt="Ancient atmospheric stone city ruins on surface under stormy dramatic sky",
                    search_query="ruins landscape",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=7,
                    narration="क्या कोई आसमानी तबाही आई थी, या फिर किसी अज्ञात दुश्मन का खौफनाक हमला?",
                    display_text="Kya koi aasmani tabahi thi ya kisi dushman ka khaufnaak hamla?",
                    visual_prompt="Ancient apocalyptic painting depicting fiery skies and fleeing shadows",
                    search_query="historical wall",
                    estimated_duration=3.9,
                ),
                ScenePrompt(
                    scene_id=8,
                    narration="इतनी विशाल इंजीनियरिंग आज के मॉडर्न इंजीनियर्स को भी पूरी तरह हैरान कर देती है।",
                    display_text="Itni vishal engineering modern engineers ko bhi heran kar deti hai.",
                    visual_prompt="Top-down geometric architectural blueprint map of multilevel underground labyrinth",
                    search_query="old architecture",
                    estimated_duration=4.0,
                ),
                ScenePrompt(
                    scene_id=9,
                    narration="क्या हमारे पूर्वज हमसे ज्यादा एडवांस थे? कमेंट्स में अपनी राय जरूर बताइए।",
                    display_text="Kya purvaj humse zyada advance the? Comments mein zaroor batayein.",
                    visual_prompt="Mysterious lone explorer looking out from a dark cave opening toward sunrise",
                    search_query="mountain cave",
                    estimated_duration=4.2,
                ),
            ]
            return ShortScript(
                topic="The Underground Citadel of Derinkuyu",
                title="Zameen Ke 85 Meter Niche Chhupa Shehar! #Shorts #History",
                description="Turkey mein zameen ke 85 meter niche 20,000 logon ka gupt shehar mila tha. Derinkuyu ka anokha rahasya! #History #Ancient #Mystery #Shorts",
                tags=["history", "ancient", "mystery", "archaeology", "shorts"],
                scenes=scenes,
            )

    # 4. Default / Psychology & Human Behavior Niche
    if is_hinglish:
        scenes = [
            ScenePrompt(
                scene_id=0,
                narration="इमेजिन कीजिए... कोई बात करते वक्त, चुपचाप आपका दिमाग स्कैन कर रहा है।",
                display_text="Imagine kijiye... koi baat karte waqt chupchap aapka dimag scan kar raha hai.",
                visual_prompt="Man in heavy shadow staring directly into camera with intense piercing gaze",
                search_query="interrogation",
                estimated_duration=4.1,
            ),
            ScenePrompt(
                scene_id=1,
                narration="इसे साइकोलॉजी में मिर्ररिंग ट्रैप कहते हैं—जो शातिर मैनिपुलेटर्स इस्तेमाल करते हैं।",
                display_text="Ise psychology mein Mirroring Trap kehte hain—jo manipulators use karte hain.",
                visual_prompt="Dramatic silhouettes across a dimly lit table in darkness",
                search_query="shadow",
                estimated_duration=4.0,
            ),
            ScenePrompt(
                scene_id=2,
                narration="वो चालाकी से आपकी बॉडी लैंग्वेज और बात करने की स्पीड कॉपी करते हैं।",
                display_text="Woh chalaki se aapki body language aur speed copy karte hain.",
                visual_prompt="Two people sitting close in deep conversation mirroring postures",
                search_query="people talking",
                estimated_duration=3.9,
            ),
            ScenePrompt(
                scene_id=3,
                narration="नतीजा? आपका सबकॉन्शियस माइंड उनपर अंधा भरोसा करने लगता है।",
                display_text="Nateeja? Aapka subconscious mind unpar andha bharosa karne lagta hai.",
                visual_prompt="Extreme macro close-up of a human eye pupil dilating under dramatic lighting",
                search_query="human eye",
                estimated_duration=3.8,
            ),
            ScenePrompt(
                scene_id=4,
                narration="सबसे खौफनाक बात... आप अपने गहरे सीक्रेट्स खुद ही खोल देते हो।",
                display_text="Sabse khaufnaak baat... aap apne gehre secrets khud hi khol dete ho.",
                visual_prompt="Mysterious dim room with a lone light bulb turning on and off",
                search_query="dark room",
                estimated_duration=4.0,
            ),
            ScenePrompt(
                scene_id=5,
                narration="लेकिन इस खतरनाक गेम को आप एक सेकंड में तोड़ सकते हो।",
                display_text="Lekin is khatarnaak game ko aap ek second mein tod sakte ho.",
                visual_prompt="Chessboard on dark oak wood table with dramatic king and pawn",
                search_query="chess",
                estimated_duration=3.9,
            ),
            ScenePrompt(
                scene_id=6,
                narration="जब भी शक हो, अपनी बॉडी लैंग्वेज तुरंत बदलो—जैसे दूर देखना।",
                display_text="Jab bhi shak ho, apni body language turant badlo—jaise door dekhna.",
                visual_prompt="Person breaking intense eye contact calmly looking to the side",
                search_query="calm person",
                estimated_duration=3.9,
            ),
            ScenePrompt(
                scene_id=7,
                narration="आप नोटिस करोगे... सामने वाला इंसान अचानक कन्फ्यूज और नर्वस हो जाएगा।",
                display_text="Aap notice karoge... saamne wala insaan achanak nervous ho jayega!",
                visual_prompt="Close up of a tense face reacting with sudden shock",
                search_query="shocked",
                estimated_duration=3.8,
            ),
            ScenePrompt(
                scene_id=8,
                narration="उनका सारा फेक कॉन्फिडेंस एक पल में हवा हो जाएगा।",
                display_text="Unka saara fake confidence ek pal mein hawa ho jayega.",
                visual_prompt="Mystical smoke swirling in darkness fading away",
                search_query="smoke",
                estimated_duration=3.7,
            ),
            ScenePrompt(
                scene_id=9,
                narration="क्या आपके साथ ऐसा खेल हुआ है? कमेंट्स में सच बताना।",
                display_text="Kya aapke sath aisa khel hua hai? Comments mein sach batana!",
                visual_prompt="Mysterious solitary figure walking down a dark street at night",
                search_query="walking",
                estimated_duration=4.2,
            ),
        ]
        return ShortScript(
            topic="The Secret Mirroring Trap & Dark Psychology",
            title="Aapka Dimaag Hack Ho Raha Hai! #Shorts #Psychology",
            description="Dark psychology mirroring trap jo log aap par secretly use karte hain. Bachne ka secret tarika! #Psychology #Shorts #Mindset #Secrets",
            tags=["psychology", "shorts", "mindset", "darkpsychology", "facts", "secrets"],
            scenes=scenes,
        )

    scenes = [
        ScenePrompt(
            scene_id=0,
            narration="If someone holds eye contact for over six seconds without smiling, beware.",
            visual_prompt="Man in heavy shadow staring directly into lens with piercing intense gaze",
            estimated_duration=4.1,
        ),
        ScenePrompt(
            scene_id=1,
            narration="Psychologists reveal this signals one of two extremes: intense attraction, or hostile dominance.",
            visual_prompt="Silhouette of two figures across a smoky dimly lit interrogation table",
            estimated_duration=4.0,
        ),
        ScenePrompt(
            scene_id=2,
            narration="When humans feel threatened, their pupils dilate slightly before their expression changes.",
            visual_prompt="Extreme macro close-up of a human eye pupil dilating against cold blue ambient light",
            estimated_duration=3.8,
        ),
        ScenePrompt(
            scene_id=3,
            narration="Dark empaths exploit this by faking vulnerability to disarm your personal boundaries.",
            visual_prompt="Subtle smirk half-covered in darkness, holding a fractured antique porcelain mask",
            estimated_duration=4.0,
        ),
        ScenePrompt(
            scene_id=4,
            narration="They mirror your posture, nod rhythmically, and subtly copy your breathing tempo.",
            visual_prompt="Two shadowy figures sitting side by side matching precise body postures like mirrors",
            estimated_duration=3.9,
        ),
        ScenePrompt(
            scene_id=5,
            narration="This floods your brain with dopamine, tricking your subconscious into trusting them.",
            visual_prompt="Glowing neurological brain synapses pulsing with electric amber dopamine sparks",
            estimated_duration=4.0,
        ),
        ScenePrompt(
            scene_id=6,
            narration="To break this spell instantly, shift your gaze directly to their ear.",
            visual_prompt="Man breaking eye gaze calmly and tilting head toward the side with cool confidence",
            estimated_duration=3.9,
        ),
        ScenePrompt(
            scene_id=7,
            narration="This unexpected redirection shatters their subconscious control and dominance immediately.",
            visual_prompt="Chessboard on a dark polished oak table where a black pawn blocks a towering white king",
            estimated_duration=3.9,
        ),
        ScenePrompt(
            scene_id=8,
            narration="Notice how quickly they blink once the psychological illusion breaks.",
            visual_prompt="Close up of a shocked face blinking rapidly under flickering warm fluorescent streetlamps",
            estimated_duration=3.7,
        ),
        ScenePrompt(
            scene_id=9,
            narration="Have you ever sensed someone manipulating your focus like this? Stay alert.",
            visual_prompt="Solitary figure walking away down a wet midnight alleyway surrounded by street reflections",
            estimated_duration=4.4,
        ),
    ]

    return ShortScript(
        topic="The 6-Second Eye Contact Trick & Dark Empath Mirroring",
        title="The 6-Second Eye Contact Danger #Shorts #Psychology",
        description="Psychologists reveal what prolonged unblinking eye contact really means and how dark empaths manipulate your trust. #DarkPsychology #Behavior #Facts #Shorts",
        tags=["psychology", "darkpsychology", "bodylanguage", "mindset", "shorts", "humanbehavior"],
        scenes=scenes,
    )


class ScriptGenerator:
    """Manages Gemini API client communication and script generation."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model or config.GEMINI_MODEL
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize Google GenAI or GenerativeAI client."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            return None

        # Try official google-genai SDK first
        try:
            from google import genai
            self._client = ("genai", genai.Client(api_key=self.api_key))
            logger.info("Initialized Google GenAI SDK (google.genai)")
            return self._client
        except (ImportError, Exception) as e:
            logger.debug("google.genai SDK not available or failed: %s. Trying google.generativeai.", e)

        # Fallback to legacy google.generativeai SDK
        try:
            import google.generativeai as gai
            gai.configure(api_key=self.api_key)
            self._client = ("legacy", gai.GenerativeModel(self.model_name))
            logger.info("Initialized legacy Google GenerativeAI SDK (google.generativeai)")
            return self._client
        except Exception as e:
            logger.warning("Could not initialize Gemini SDK: %s", e)
            return None

    def _call_gemini_api(self, prompt: str, system_instruction: str) -> str:
        """
        Execute Gemini API call with automatic multi-model cascading fallback.
        Tries primary model, then cascades through available flash-lite and flash models
        if quota (HTTP 429), temporary unavailability (HTTP 503), or deprecated model (HTTP 404) occurs.
        """
        candidate_models = [
            self.model_name,
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
        ]
        unique_models = list(dict.fromkeys([m for m in candidate_models if m]))
        last_error: Exception | None = None

        for model_to_try in unique_models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_try}:generateContent?key={self.api_key}"
                payload = {
                    "system_instruction": {"parts": [{"text": system_instruction}]},
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.75,
                        "responseMimeType": "application/json",
                    },
                }
                res = requests.post(url, json=payload, timeout=45)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = parts[0].get("text", "") if parts else ""
                        if text.strip():
                            if self.model_name != model_to_try:
                                logger.info("Model '%s' succeeded. Upgraded active model.", model_to_try)
                            self.model_name = model_to_try
                            return text
                    raise RuntimeError(f"No candidate text returned by {model_to_try}")
                else:
                    raise RuntimeError(f"HTTP {res.status_code}: {res.text[:140]}")

            except Exception as e:
                err_str = str(e)
                logger.warning("Gemini model '%s' failed (%s). Cascading to next model...", model_to_try, err_str[:120])
                last_error = e
                continue

        raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")

    def generate_script(self, custom_topic: str | None = None) -> ShortScript:
        """Generate a brand-new structured Short script."""
        recent_topics = load_recent_topics(max_history=40)
        system_instruction = _build_system_prompt(recent_topics, config.NICHE_TOPIC)

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Using offline fallback script.")
            return generate_fallback_offline_script(config.NICHE_TOPIC)

        user_prompt = f"Generate an ultra-engaging, viral YouTube Short in the niche '{config.NICHE_TOPIC}'."
        if custom_topic:
            user_prompt += f" Focus specifically on the topic: '{custom_topic}'."
        else:
            user_prompt += f" Pick a shocking, mind-blowing, little-known breakthrough fact or mystery in '{config.NICHE_TOPIC}' that triggers extreme curiosity and sharing."

        user_prompt += f"\nREMINDER: Total spoken word count across ALL scenes must strictly be between {config.TARGET_MIN_WORDS} and {config.TARGET_MAX_WORDS} words."

        logger.info("Requesting script from Gemini for niche '%s' using model '%s'...", config.NICHE_TOPIC, self.model_name)
        try:
            raw_text = self._call_gemini_api(user_prompt, system_instruction)
            script = _parse_gemini_response(raw_text)
            logger.info(
                "Successfully generated script: '%s' (%d scenes, %d words)",
                script.topic,
                len(script.scenes),
                script.total_words,
            )
            return script
        except Exception as e:
            logger.error("Gemini generation failed (%s). Falling back to curated script for niche '%s'.", e, config.NICHE_TOPIC)
            return generate_fallback_offline_script(config.NICHE_TOPIC)

    def re_prompt_with_duration_feedback(
        self,
        previous_script: ShortScript,
        actual_duration: float,
        target_range: tuple[float, float] = (40.0, 45.0),
    ) -> ShortScript:
        """
        Re-prompts Gemini with explicit expansion or compression guidance
        when measured audio duration falls outside the strict 40.0-45.0s window.
        """
        min_dur, max_dur = target_range
        total_words = previous_script.total_words
        # Average speaking speed is approx 2.65 words/second
        avg_wps = total_words / max(actual_duration, 1.0)

        if actual_duration < min_dur:
            deficit_seconds = min_dur - actual_duration
            words_to_add = max(int(deficit_seconds * avg_wps) + 4, 6)
            adjustment_type = "EXPAND"
            feedback_msg = (
                f"Your previous script generated {actual_duration:.2f} seconds of spoken audio ({total_words} words), "
                f"which is TOO SHORT (minimum target is {min_dur:.1f}s). "
                f"You MUST expand the narration text by adding approximately {words_to_add} to {words_to_add + 6} words "
                f"across the existing {len(previous_script.scenes)} scenes. Add richer atmospheric descriptors, "
                f"psychological nuance, and dramatic tension while keeping the exact same topic and scene structure."
            )
        else:
            excess_seconds = actual_duration - max_dur
            words_to_trim = max(int(excess_seconds * avg_wps) + 3, 5)
            adjustment_type = "COMPRESS"
            feedback_msg = (
                f"Your previous script generated {actual_duration:.2f} seconds of spoken audio ({total_words} words), "
                f"which is TOO LONG (maximum target is {max_dur:.1f}s). "
                f"You MUST condense the narration text by cutting approximately {words_to_trim} words "
                f"across the existing {len(previous_script.scenes)} scenes. Make every single sentence tighter, "
                f"punchier, and faster-paced while maintaining the core revelation and retention hooks."
            )

        logger.info(
            "Duration feedback triggered (%s): Audio took %.2fs (target: %.1f-%.1fs). Re-prompting Gemini...",
            adjustment_type,
            actual_duration,
            min_dur,
            max_dur,
        )

        if not self.api_key:
            # If no API key, programmatically adjust word count on fallback
            logger.info("API key unset during re-prompt; returning previous script.")
            return previous_script

        recent_topics = load_recent_topics(max_history=40)
        system_instruction = _build_system_prompt(recent_topics, config.NICHE_TOPIC)

        reprompt_prompt = f"""{feedback_msg}

PREVIOUS SCRIPT JSON TO REVISE:
{previous_script.model_dump_json(indent=2)}

Respond ONLY with the revised raw JSON object with the adjusted narrations."""

        try:
            raw_text = self._call_gemini_api(reprompt_prompt, system_instruction)
            revised_script = _parse_gemini_response(raw_text)
            logger.info(
                "Revised script generated: %d scenes, %d words (was %d words)",
                len(revised_script.scenes),
                revised_script.total_words,
                previous_script.total_words,
            )
            return revised_script
        except Exception as e:
            logger.warning("Re-prompting failed (%s). Continuing with existing script.", e)
            return previous_script
