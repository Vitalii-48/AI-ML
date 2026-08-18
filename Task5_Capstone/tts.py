import pyttsx3
def speak(text: str) -> None:
    """Converts text to speech using local offline pyttsx3 engine."""
    if not text:
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"\n[TTS Error]: Could not play audio via pyttsx3: {e}")