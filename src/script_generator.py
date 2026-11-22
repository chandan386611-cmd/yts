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
            "LANGUAGE REQUIREMENT (CRITICAL FOR NATURAL HUMAN PRONUNCIATION):\n"
            "1. 'narration': Spoken voiceover text. MUST be written in conversational HINDI (Devanagari script, e.g. 'क्या आपने कभी सोचा है...', 'पिरामिड्स के पीछे का असली सच...'). "
            "You can include common English loan words in English letters (e.g., 'Pyramids', 'Secret', 'Technology', 'Science'). Devanagari guarantees the neural voice engine speaks with 100% natural, human, authentic Indian pronunciation with ZERO robotic glitches!\n"
            "2. 'display_text': On-screen subtitle text. MUST be written in punchy Romanized HINGLISH (Hindi in English/Latin letters, e.g., 'Kya aapne kabhi socha hai...', 'Pyramids ka asli sach...'). This will be rendered as bold center-screen pop-in subtitles.\n"
            "3. 'title' & 'description': Write in catchy, high-CTR Hinglish with relevant hashtags.\n"
            "4. 'visual_prompt': MUST ALWAYS BE IN ENGLISH for the AI image generator."
        )
    else:
        language_rule = (
            "LANGUAGE: All narration, display_text, title, and description MUST be written in punchy, captivating English. "
            "Visual prompts must be in English."
        )

    return f"""You are an elite YouTube Shorts scriptwriter and retention psychologist specializing in viral vertical video content in the niche: '{niche}'.

CRITICAL RETENTION & TIMING RULES:
1. {language_rule}
2. TOTAL SPOKEN WORDS: MUST BE STRICTLY BETWEEN {config.TARGET_MIN_WORDS} AND {config.TARGET_MAX_WORDS} WORDS TOTAL across the entire script. Spoken at an energetic brisk pace (150-165 WPM), this results in exactly 40.0 to 45.0 seconds of audio.
3. SCENE COUNT: EXACTLY {config.MIN_SCENES} TO {config.MAX_SCENES} DISTINCT SCENES (numbered 0 to N-1).
   - Scene 0 is THE 5-SECOND OPENING HOOK. It must ignite instant curiosity or fear of missing out, shocking the viewer in under 12 words without intro fluff.
   - Scenes 1 to N-1 must develop the psychological mechanism, dark behavioral revelation, or counter-intuitive truth.
   - Scene N-1 must deliver a punchy payoff/takeaway and a subtle engagement cliffhanger (e.g., 'Aapko kya lagta hai? Comments mein batao.').
4. PACING: Each individual scene narration must be 9 to 14 words (~3.5 to 4.2 seconds).
5. VISUAL PROMPTS: Each scene MUST have a highly atmospheric visual description in English focusing on symbolic metaphors, intense facial micro-expressions, shadows, retro-modern dark rooms, or surreal motifs. Do NOT include words like 'cinematic' or 'hyper-detailed' in visual_prompt—art style will be locked automatically.
6. STRICT DEDUPLICATION: DO NOT reuse or closely duplicate any of the following recently covered topics:
{avoid_list if avoid_list else "(No prior topics - pick an ultra-viral breakthrough fact)"}

OUTPUT FORMAT:
You MUST respond ONLY with a valid raw JSON object matching this exact structure:
{{
  "topic": "Concise topic title",
  "title": "Viral Hook Title Under 60 Chars #Shorts #History",
  "description": "Engaging 2-3 sentence description with high SEO keywords and hashtags #History #Shorts #Mysteries",
  "tags": ["history", "mysteries", "shorts", "conspiracy", "facts"],
  "scenes": [
    {{
      "scene_id": 0,
      "narration": "क्या आपने कभी सोचा है कि इजिप्ट के Pyramids सच में किसने बनाए? (Spoken Hindi for TTS)",
      "display_text": "Kya aapne kabhi socha hai ki Egypt ke Pyramids kisne banaye? (Romanized Hinglish for subtitles)",
      "visual_prompt": "Subject and action description without style adjectives in English",
      "search_query": "2-3 clean keywords for real photographic search (e.g., 'Pyramids of Giza')",
      "estimated_duration": 4.0
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
                narration="Agar koi aapse baat karte waqt 6 second se zyada bina muskuraye dekhe, to savdhan ho jao.",
                visual_prompt="Man in heavy shadow staring directly into lens with piercing intense gaze",
                estimated_duration=4.1,
            ),
            ScenePrompt(
                scene_id=1,
                narration="Psychology kehti hai ki iske do hi matlab hote hain: ya to intense attraction, ya hostile dominance.",
                visual_prompt="Silhouette of two figures across a smoky dimly lit interrogation table",
                estimated_duration=4.0,
            ),
            ScenePrompt(
                scene_id=2,
                narration="Jab insaan insecure ya threatened feel karta hai, to uski aankhein secretly dilate hone lagti hain.",
                visual_prompt="Extreme macro close-up of a human eye pupil dilating against cold blue ambient light",
                estimated_duration=3.8,
            ),
            ScenePrompt(
                scene_id=3,
                narration="Dark empaths isi cheez ka fayda uthate hain, aapse jhuth bolkar aapka trust jeetne ke liye.",
                visual_prompt="Subtle smirk half-covered in darkness, holding a fractured antique porcelain mask",
                estimated_duration=4.0,
            ),
            ScenePrompt(
                scene_id=4,
                narration="Woh aapke body posture aur baat karne ke style ko secretly copy karna shuru karte hain.",
                visual_prompt="Two shadowy figures sitting side by side matching precise body postures like mirrors",
                estimated_duration=3.9,
            ),
            ScenePrompt(
                scene_id=5,
                narration="Isse aapke dimag mein dopamine release hota hai aur aap bina soche unpar bharosa karne lagte ho.",
                visual_prompt="Glowing neurological brain synapses pulsing with electric amber dopamine sparks",
                estimated_duration=4.0,
            ),
            ScenePrompt(
                scene_id=6,
                narration="Is psychological trap ko todne ke liye, unki aankhon ke bajaye unke kaan ki taraf dekho.",
                visual_prompt="Man breaking eye gaze calmly and tilting head toward the side with cool confidence",
                estimated_duration=3.9,
            ),
            ScenePrompt(
                scene_id=7,
                narration="Aisa karne se unka mental control turant break ho jayega aur woh uncomfortable feel karenge.",
                visual_prompt="Chessboard on a dark polished oak table where a black pawn blocks a towering white king",
                estimated_duration=3.9,
            ),
            ScenePrompt(
                scene_id=8,
                narration="Notice karna, jaise hi aap ye karoge, unka fake confidence ek second mein gayab ho jayega.",
                visual_prompt="Close up of a shocked face blinking rapidly under flickering warm fluorescent streetlamps",
                estimated_duration=3.7,
            ),
            ScenePrompt(
                scene_id=9,
                narration="Kya aapke sath kabhi kisi ne aisa manipulation kiya hai? Comments mein zarur batana.",
                visual_prompt="Solitary figure walking away down a wet midnight alleyway surrounded by street reflections",
                estimated_duration=4.4,
            ),
        ]
        return ShortScript(
            topic="6-Second Eye Contact Trick & Dark Empath Mirroring",
            title="Yeh Psychological Trick Aapko Bachayegi! #Shorts #Psychology",
            description="Dark psychology trick jo aapko manipulation se bachayegi. Watch till end! #Psychology #Shorts #Facts #HumanBehavior",
            tags=["psychology", "shorts", "mindset", "darkpsychology", "facts"],
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
