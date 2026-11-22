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

    return f"""You are an elite Hollywood documentary filmmaker and viral YouTube Shorts thriller writer in the niche: '{niche}'.
Your scripts do NOT sound like dry textbooks, Wikipedia articles, or academic lectures. Instead, you craft intense, addictive micro-thrillers inspired by Christopher Nolan and high-retention psychological mysteries.

CRITICAL 4-BEAT THRILLER STORYTELLING ARC:
1. Beat 1: The Shocking Cold Open (Scene 0)
   - Under 13 words. Shatter common sense or warn the viewer about an invisible danger (e.g. 'इमेजिन कीजिए... कोई बात करते वक्त, चुपचाप आपका दिमाग स्कैन कर रहा है।').
   - ZERO fluff, no 'Hello friends' or intro titles.
2. Beat 2: The Deepening Mystery & The Hidden Trap (Scenes 1 to 3)
   - Unfold the forbidden psychological mechanism or classified historical experiment with sensory tension.
3. Beat 3: The Chilling Twist & Biological Vulnerability (Scenes 4 to 6)
   - The mind-blowing realization: this is happening to the viewer in their daily life without their consent.
4. Beat 4: The 1-Second Countermeasure & Comment Trigger (Scenes 7 to N-1)
   - Give the viewer an instant psychological shield, ending on a cliffhanger question that forces them to comment and debate.

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
  "title": "Viral Hook Title Under 60 Chars #Shorts #Psychology",
  "description": "Engaging 2-3 sentence description with high SEO keywords and hashtags #Psychology #Shorts #Mysteries",
  "tags": ["psychology", "mindset", "shorts", "darkpsychology", "facts"],
  "scenes": [
    {{
      "scene_id": 0,
      "narration": "इमेजिन कीजिए... कोई बात करते वक्त, चुपचाप आपका दिमाग स्कैन कर रहा है।",
      "display_text": "Imagine kijiye... koi baat karte waqt chupchap aapka dimag scan kar raha hai.",
      "visual_prompt": "Man in heavy shadow staring directly into camera with intense piercing gaze",
      "search_query": "interrogation",
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
    script = ShortScript(
        topic=data["topic"],
        title=data["title"],
        description=data["description"],
        tags=data.get("tags", ["psychology", "shorts"]),
        scenes=scenes,
    )
    return script


def generate_fallback_offline_script(niche: str) -> ShortScript:
    """Generate a high-quality offline script if Gemini API key is unset or in dry-run mode."""
    logger.info("Generating curated offline fallback script for niche: %s", niche)
    is_hinglish = getattr(config, "LANGUAGE", "hinglish").lower() in ("hinglish", "hindi")

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_gemini_api(self, prompt: str, system_instruction: str) -> str:
        """Execute Gemini API call with exponential backoff (SDK or direct REST)."""
        client_bundle = self._get_client()

        if client_bundle is None:
            # Direct REST API fallback using requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.75,
                    "responseMimeType": "application/json",
                },
            }
            res = requests.post(url, json=payload, timeout=45)
            if res.status_code != 200:
                raise RuntimeError(f"Gemini API error (HTTP {res.status_code}): {res.text}")
            data = res.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"No candidates returned by Gemini: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else ""

        sdk_type, client = client_bundle

        if sdk_type == "genai":
            from google.genai import types
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.75,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""
        else:
            # Legacy google.generativeai
            full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_instruction}\n\nUSER PROMPT:\n{prompt}"
            response = client.generate_content(
                full_prompt,
                generation_config={"temperature": 0.75, "response_mime_type": "application/json"},
            )
            return response.text or ""

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
            user_prompt += " Pick a shocking, little-known psychological phenomenon or behavioral bias that triggers high curiosity and sharing."

        user_prompt += f"\nREMINDER: Total spoken word count across ALL scenes must strictly be between {config.TARGET_MIN_WORDS} and {config.TARGET_MAX_WORDS} words."

        logger.info("Requesting script from Gemini model '%s'...", self.model_name)
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
            logger.error("Gemini generation failed (%s). Falling back to curated script.", e)
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
