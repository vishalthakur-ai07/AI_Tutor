import json
import os
import re

from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

app = Flask(__name__, static_folder="static")

# NVIDIA's hosted models use the OpenAI-style API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY", "missing-key"),
)
MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

# ---------- STEP 1: the syllabus (CBSE Class 8, NCERT Ganita Prakash) ----------
CHAPTERS = {
    "Numbers and powers": ["P1-1 A Square and A Cube", "P1-2 Power Play",
                           "P1-3 A Story of Numbers", "P1-5 Number Play"],
    "Algebra": ["P1-6 We Distribute, Yet Things Multiply", "P2-6 Algebra Play"],
    "Ratio, percentage and proportion": ["P2-1 Fractions in Disguise",
                                         "P1-7 Proportional Reasoning-1",
                                         "P2-3 Proportional Reasoning-2"],
    "Geometry": ["P1-4 Quadrilaterals", "P2-2 The Baudhayana-Pythagoras Theorem",
                 "P2-4 Exploring Some Geometric Themes"],
    "Mensuration": ["P2-7 Area"],
    "Data": ["P2-5 Tales by Dots and Lines"],
}

PREREQUISITES = (
    "P2-2 needs P1-1. P2-7 needs P1-4 and P2-2. P2-6 needs P1-6. "
    "P2-3 needs P1-7, and P2-1 supports both. P2-5 needs a basic grasp of ratios and percentages."
)

# ---------- STEP 2: the tutor's personality ----------
SYSTEM_TEMPLATE = """You are MathMentor, a friendly and patient mathematics tutor for CBSE Class 8 students in India. You follow the NCERT textbook "Ganita Prakash" (Part 1 and Part 2) and use its terminology and way of introducing ideas.

SCOPE
You teach only these 14 chapters, grouped into six units:
{SCOPE}
If a student asks about another class, another subject, or anything unrelated to these chapters, say kindly that you only help with Class 8 maths, and offer a related chapter instead. Do not answer off-topic questions.

PERSONALITY
- Be warm, patient, and encouraging. Never be sarcastic, impatient, or condescending.
- Use simple, everyday language. The first time you use a maths term, explain it in plain words.
- Keep replies short: about 3 to 6 sentences, plus one question or task for the student.
- Praise specific effort ("You spotted the common factor") rather than saying only "Good job!".

HOW YOU TEACH
- Explain one idea at a time, using a real-life example (money, cricket, food, distances) and one short worked example with each step shown.
- After explaining, ask the student to try a similar problem. Do not move on until they have attempted it.
- Ask questions often, such as "What do you think the first step is?", so the student does the thinking.
- After a few exchanges on a topic, offer a short quiz of 3 questions.
- If a student is weak in a chapter that depends on an earlier one, suggest revising the earlier chapter first. Dependencies: {PREREQS}

HINTS BEFORE ANSWERS
When a student is stuck or gets something wrong, use this ladder and go down only if needed:
1. A gentle nudge. 2. Name the method or first step. 3. Work the first step and ask them to continue. 4. Show the full solution with reasons, then give a similar problem.
Do not give the final answer until the student has tried at least twice, unless they have already attempted it and ask to see the solution.

CHECKING ANSWERS
- Before judging a student's answer, work out the correct answer yourself, step by step.
- If correct, say so and explain what they did well. If wrong, find the exact step where the mistake happened, explain it kindly, and invite them to try again.
- If you are not sure of something, say so honestly and work it out step by step instead of guessing.

PICTURES
You cannot show diagrams. For geometry, area, and graph topics, describe figures in words, put all measurements in the question, and avoid tasks like "draw this".

FORMAT
Write maths in plain text, such as 3/4, x^2, or 2x + 5 = 11. Do not do a whole homework in one go. End every reply with exactly one clear next step: a question, a problem to try, or a choice."""


def scope_text():
    return "\n".join(
        f"{i}. {unit}: " + "; ".join(chs) for i, (unit, chs) in enumerate(CHAPTERS.items(), 1)
    )


SYSTEM_PROMPT = SYSTEM_TEMPLATE.replace("{SCOPE}", scope_text()).replace("{PREREQS}", PREREQUISITES)

# ---------- STEP 4: modes that use normal chat ----------
MODE_TEXT = {
    "learn": "LEARN MODE: Explain the current chapter step by step in small pieces. Start with a simple real-life example, then one worked example, then check understanding with ONE short question. No long lectures and no quiz.",
    "doubt": "ASK A DOUBT MODE: The student will describe something they do not understand or a problem they are stuck on. Find out exactly what confuses them (ask one clarifying question if needed), clear it with a simple explanation and example, and guide with hints instead of solving everything. Finish by checking they understood.",
}

# ---------- STEP 5: personalization ----------
DIFFICULTY = {
    "Beginner": "Beginner: use very simple language, plenty of everyday examples, small steps and easy problems. Offer hints early and generously, and encourage often.",
    "Intermediate": "Intermediate: use clear explanations with moderate problems. Follow the normal hint ladder and keep a balanced pace.",
    "Advanced": "Advanced: keep explanations concise and use challenging multi-step problems. Give fewer hints, only after a real attempt, and ask the student to justify their reasoning.",
}


def profile_text(p):
    diff = p.get("difficulty", "Beginner")
    return (
        f"STUDENT PROFILE: name {p.get('name', 'the student')}; "
        f"how they feel about maths: {p.get('feeling', 'okay')}; difficulty preference: {diff}. "
        f"{DIFFICULTY.get(diff, DIFFICULTY['Beginner'])} Address the student by name occasionally."
    )


# ---------- talking to the NVIDIA model ----------
def call(system, messages, max_tokens=700, temperature=0.5):
    full = [{"role": "system", "content": system}] + messages
    r = client.chat.completions.create(
        model=MODEL, messages=full, max_tokens=max_tokens, temperature=temperature
    )
    text = r.choices[0].message.content or ""
    # some models print their reasoning inside <think> tags; hide it from students
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text):
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        raise ValueError("No JSON found in the AI reply")
    return json.loads(m.group(0))


def call_json(system, prompt, max_tokens, temperature=0.2):
    """Ask for JSON; if the model's reply cannot be read, try once more."""
    last = None
    for _ in range(2):
        try:
            return parse_json(call(system, [{"role": "user", "content": prompt}], max_tokens, temperature))
        except Exception as e:
            last = e
    raise last


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def config():
    return jsonify({"chapters": CHAPTERS})


# ---------- STEP 3 and 4: chat (Learn and Ask a Doubt) ----------
@app.route("/api/chat", methods=["POST"])
def chat():
    d = request.get_json(force=True)
    messages = d.get("messages", [])
    if not messages or messages[-1].get("role") != "user":
        return jsonify({"error": "messages must end with a user message"}), 400
    system = (
        SYSTEM_PROMPT + "\n\n" + MODE_TEXT.get(d.get("mode"), MODE_TEXT["learn"])
        + f"\n\nCURRENT CHAPTER: {d.get('chapter', 'any chapter')}\n" + profile_text(d.get("profile", {}))
    )
    try:
        return jsonify({"reply": call(system, messages)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- STEP 6: question generation (nothing is stored) ----------
@app.route("/api/questions", methods=["POST"])
def questions():
    d = request.get_json(force=True)
    chapter = d.get("chapter", "")
    level = d.get("difficulty", "Beginner")
    count = max(1, min(int(d.get("count", 3)), 8))
    avoid = d.get("avoid", [])
    styles = {
        "Beginner": "easy, one or two steps, small whole numbers",
        "Intermediate": "moderate, two or three steps, may include simple word problems",
        "Advanced": "challenging, needing three or four steps, may be a word problem; stated in at most three short sentences",
    }
    schema = ('[{"question": "A rectangle has length 12 m and width 5 m. What is its area?", '
              '"answer": "60 square metres", "hint": "Multiply the length by the width.", '
              '"concept": "area of a rectangle"}]')
    prompt = (
        f"Create {count} NEW questions for CBSE Class 8 maths, NCERT Ganita Prakash chapter: {chapter}.\n"
        f"Difficulty: {level} ({styles.get(level, styles['Beginner'])}).\n"
        "Rules: each question must be answerable in text with ONE short final answer (a number, expression or short phrase). "
        "Put all numbers and measurements inside the question. Never ask the student to draw. Double-check every answer. "
        "Write maths in plain text only (3/4, x^2, 2x + 5 = 11). Do NOT use LaTeX, backslashes or dollar signs. "
        "The hint must help without giving the answer away. 'concept' is the specific skill tested.\n"
        f"Do not repeat or closely copy these earlier questions: {json.dumps(avoid)}\n"
        "Reply with ONLY a JSON list. The example below shows the SHAPE only; your questions must be different and about the chapter above:\n"
        f"{schema}"
    )
    try:
        qs = call_json("You write accurate maths questions. Reply with JSON only.", prompt, 3000)
        if isinstance(qs, dict):
            qs = [qs]
        good = [
            q for q in qs
            if isinstance(q, dict) and all(k in q for k in ("question", "answer", "hint", "concept"))
            and len(str(q["question"]).strip()) > 15 and str(q["answer"]).strip() not in ("", "...")
        ]
        if not good:
            raise ValueError("No valid questions came back")
        return jsonify({"questions": good})
    except Exception as e:
        print("Question generation failed:", e, flush=True)
        return jsonify({"error": str(e)}), 500


# ---------- STEP 7: answer evaluation ----------
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    d = request.get_json(force=True)
    reveal = bool(d.get("reveal", True))
    schema = ('{"verdict": "correct" or "almost" or "incorrect", "correct_answer": "...", '
              '"explanation": "...", "hint": "...", "encouragement": "..."}')
    if reveal:
        how = "Give a short, kind, step-by-step explanation in 'explanation'."
    else:
        how = ("The student gets another try, so do NOT reveal the final answer or the full solution. "
               "In 'explanation' only point to where the mistake probably is.")
    prompt = (
        f"Chapter: {d.get('chapter')}\nQuestion: {d.get('question')}\n"
        f"Reference answer: {d.get('answer')}\nStudent's answer: {d.get('student')}\n\n"
        "First solve the question yourself step by step. The reference answer may rarely be wrong; "
        "if your own working disagrees with it, trust your working. Accept equivalent forms (for example 0.5 and 1/2, or x = 6 and 6). "
        "Use 'almost' when the method is right but a small slip was made. "
        f"{how} 'hint' is one hint that does not give the answer away. 'encouragement' is one short, warm sentence.\n"
        f"Reply with ONLY JSON in exactly this shape: {schema}"
    )
    system = "You are MathMentor, a kind Class 8 maths tutor. Reply with JSON only. " + profile_text(d.get("profile", {}))
    try:
        result = call_json(system, prompt, 800)
        for key in ("verdict", "correct_answer", "explanation", "hint", "encouragement"):
            result.setdefault(key, "")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- STEP 9: end-of-session recommendation ----------
@app.route("/api/summary", methods=["POST"])
def summary():
    d = request.get_json(force=True)
    stats = d.get("stats", {})
    attempted = sum(s["attempted"] for s in stats.values())
    correct = sum(s["correct"] for s in stats.values())
    prompt = (
        f"Student: {d.get('profile', {}).get('name', 'the student')}\n"
        f"Questions attempted: {attempted}\nCorrect: {correct}\n"
        f"Results by chapter: {json.dumps(stats)}\nSkills the student got wrong: {json.dumps(d.get('weak', {}))}\n"
        f"Allowed chapters: {json.dumps(sum(CHAPTERS.values(), []))}\nDependencies: {PREREQUISITES}\n\n"
        "Write a short session summary in plain text using EXACTLY this layout:\n"
        "Today's Performance\nQuestions attempted: <number>\nCorrect: <number>\nStrong areas: <bullet list>\n"
        "Needs improvement: <bullet list of skills or chapters>\n"
        "Recommended next lesson: <ONE chapter from the allowed list, chosen using the weak areas and dependencies, with a one-line reason>\n"
        "Then one warm sentence of encouragement. Copy the two numbers exactly."
    )
    try:
        return jsonify({"summary": call("You are MathMentor, a kind Class 8 maths tutor.",
                                        [{"role": "user", "content": prompt}], 600, 0.3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not os.environ.get("NVIDIA_API_KEY"):
        print("WARNING: NVIDIA_API_KEY is not set. Set it before using the tutor.")
    app.run(debug=True, port=5000)