"""Small seeded bilingual dialogue fixtures, not a chat-quality dataset."""
import random

INTENTS = ("greeting", "sensor_state", "yes", "no", "thanks", "bye")
LANGUAGES = ("es", "en")
_PAIRS = {
    "es": {
        "greeting": (("Hola", "¡Hola!"), ("Buenos días", "Buenos días.")),
        "sensor_state": (("¿Cuál es la temperatura?", "La temperatura es {value} °C."),),
        "yes": (("El sensor está activo. ¿Está activo?", "Sí, está activo."),),
        "no": (("El sensor está apagado. ¿Está activo?", "No, está apagado."),),
        "thanks": (("Gracias", "De nada."), ("Muchas gracias", "Con gusto.")),
        "bye": (("Adiós", "¡Adiós!"), ("Hasta luego", "Hasta luego.")),
    },
    "en": {
        "greeting": (("Hello", "Hello!"), ("Good morning", "Good morning.")),
        "sensor_state": (("What is the temperature?", "The temperature is {value} °C."),),
        "yes": (("The sensor is on. Is it on?", "Yes, it is on."),),
        "no": (("The sensor is off. Is it on?", "No, it is off."),),
        "thanks": (("Thanks", "You're welcome."), ("Thank you", "My pleasure.")),
        "bye": (("Goodbye", "Goodbye!"), ("See you", "See you later.")),
    },
}


def generate(seed, per_intent=1):
    """Return shuffled records {language, intent, turns}; 12*per_intent records.

    Every language/intent occurs exactly per_intent times. Values are synthetic
    and supplied in system context, never invented as live sensor readings.
    Uses a private RNG; no files, network, or global RNG changes.
    """
    if type(seed) is not int or type(per_intent) is not int or not 1 <= per_intent <= 1000:
        raise ValueError("integer seed and per_intent in 1..1000 required")
    rng = random.Random(seed)
    records = []
    for language in LANGUAGES:
        for intent in INTENTS:
            for _ in range(per_intent):
                value = rng.randrange(15, 36)
                user, answer = rng.choice(_PAIRS[language][intent])
                system = (f"Datos simulados: temperatura = {value} °C. Responde brevemente."
                          if language == "es" else
                          f"Synthetic data: temperature = {value} °C. Answer briefly.")
                records.append({"language": language, "intent": intent,
                                "turns": [("system", system), ("user", user),
                                          ("assistant", answer.format(value=value))]})
    rng.shuffle(records)
    return records
