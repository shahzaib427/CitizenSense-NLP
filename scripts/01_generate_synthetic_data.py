"""
Step 1 - Generate the SYNTHETIC raw dataset for CitizenSense.

IMPORTANT: every sample here is artificially generated from hand-written
phrase banks (English, Roman Urdu and mixed). No real citizen data is used.
State this clearly in your report (the brief requires it).

The generator deliberately injects realistic "dirt" (URLs, phone numbers,
duplicates, extra spaces, typos, emojis, empty rows) so the cleaning step
in 02_clean_and_split.py has real work to do.

Run:  python scripts/01_generate_synthetic_data.py
Out:  data/raw/feedback_raw.csv
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Phrase banks: (english, roman_urdu, severity 0-3)
# severity: 0 minor annoyance, 1 normal problem, 2 serious, 3 danger to life
# ---------------------------------------------------------------------------
NEG = {
    "Roads / Infrastructure": [
        ("The road in our area is badly broken", "hamare area ki road bohat kharab hai", 1),
        ("There are huge potholes on the main street", "main sarak par bade bade gaddhe hain", 1),
        ("Rainwater collects on the damaged road every day", "toot-phoot road par barish ka pani jama ho jata hai", 1),
        ("The bridge near our locality has developed cracks", "hamare muhalle ke qareeb pul mein dararein par gayi hain", 2),
        ("An open manhole on the road is dangerous for children", "sarak par khula manhole bachon ke liye khatarnak hai", 3),
        ("The road was dug up for work and never repaired", "road khod kar chor di gayi aur dobara nahi banayi", 1),
        ("The street is full of gravel and dust since construction stopped", "construction ruk gayi hai aur gali mein bajri aur dhool hai", 0),
        ("The speed breaker is too high and damages vehicles", "speed breaker bohat ooncha hai gaariyan kharab hoti hain", 0),
        ("The footpath is broken and unsafe for elderly people", "footpath toota hua hai buzurgon ke liye ghair mehfooz hai", 1),
        ("The street surface has collapsed near the school", "school ke qareeb sarak dhans gayi hai", 2),
    ],
    "Water Supply": [
        ("There is no water supply in our area", "hamare area mein pani ki supply bilkul band hai", 2),
        ("The tap water is dirty and smells bad", "nalke ka pani ganda hai aur badbu aati hai", 2),
        ("Water pressure is very low in the morning", "subah pani ka pressure bohat kam hota hai", 1),
        ("The main water pipeline is leaking and wasting water", "pani ki main pipeline leak ho rahi hai aur pani zaya ho raha hai", 1),
        ("We are forced to buy tanker water", "humein tanker ka pani kharidna par raha hai", 1),
        ("The supplied water is contaminated and children are falling sick", "pani aloodah hai aur bachay beemar ho rahe hain", 3),
        ("The tube well has stopped working", "tube well kharab ho gaya hai", 2),
        ("Water comes only for ten minutes a day", "pani din mein sirf das minute aata hai", 1),
        ("The water bill is wrong although we get no supply", "pani nahi aata lekin bill phir bhi aa gaya hai", 0),
        ("The water filtration plant in our locality is not working", "hamare muhalle ka filtration plant band para hai", 1),
    ],
    "Electricity": [
        ("There is no electricity since morning", "subah se bijli nahi hai", 2),
        ("Load shedding is too long in our area", "hamare area mein load shedding bohat zyada hai", 1),
        ("The transformer blew up and was never replaced", "transformer jal gaya hai aur abhi tak nahi badla gaya", 2),
        ("Loose electric wires are hanging near the street", "bijli ki taarein gali mein latak rahi hain", 3),
        ("The street lights are not working at night", "raat ko street lights band hoti hain", 1),
        ("The electricity bill is much higher than usual", "bijli ka bill bohat zyada aya hai", 0),
        ("Voltage fluctuation is damaging our appliances", "voltage upar neeche hone se hamare appliances kharab ho rahe hain", 1),
        ("Sparks come out of the electric pole every time it rains", "barish mein bijli ke khambay se chingariyan nikalti hain", 3),
        ("Power is cut for hours without any notice", "bina bataye kai ghantay bijli band rehti hai", 1),
        ("The electric meter was not replaced after repeated requests", "baar baar kehne ke bawajood meter nahi badla gaya", 0),
    ],
    "Sanitation": [
        ("Garbage is not collected from our street", "hamari gali se kooda nahi uthaya ja raha", 1),
        ("The sewer line is overflowing onto the street", "gutter ka pani sarak par beh raha hai", 2),
        ("The drain is blocked and smells terrible", "naali band hai aur bohat badbu aa rahi hai", 1),
        ("A heap of waste has been lying near the market for days", "bazar ke paas kachray ka dher kai dinon se para hai", 1),
        ("Stagnant dirty water is breeding mosquitoes", "khara ganda pani machar paida kar raha hai", 2),
        ("There is a dead animal on the road and nobody removes it", "sarak par murda janwar para hai koi nahi uthata", 1),
        ("Sweepers do not come to our street", "jharoo dene wale hamari gali mein nahi aate", 0),
        ("Open dumping of waste beside the park is spreading disease", "park ke saath kachra phenkne se beemariyan phail rahi hain", 2),
        ("Sewage water is mixing with the drinking water line", "gutter ka pani peenay ke pani mein mil raha hai", 3),
        ("There are no dustbins in our locality", "hamare muhalle mein koi dustbin nahi hai", 0),
    ],
    "Health": [
        ("The waiting time at the hospital is too long", "hospital mein intezar ka waqt bohat zyada hai", 1),
        ("There are no medicines available at the dispensary", "dispensary mein dawaiyan maujood nahi hain", 2),
        ("The doctor is absent during duty hours", "doctor duty ke auqat mein ghair hazir rehta hai", 1),
        ("The hospital staff was rude to the patients", "hospital ka staff mareezon ke sath badtameezi karta hai", 1),
        ("The ambulance reached very late for a critical patient", "critical mareez ke liye ambulance bohat der se pohanchi", 3),
        ("The hospital wards are dirty and unhygienic", "hospital ke wards ganday aur ghair saaf hain", 1),
        ("There is no oxygen supply at the emergency ward", "emergency ward mein oxygen maujood nahi hai", 3),
        ("Dengue cases are rising and there is no spray in our area", "dengue ke case barh rahe hain aur spray nahi hui", 2),
        ("The lab machines are broken so tests are delayed", "lab ki machinein kharab hain isliye tests mein der hoti hai", 1),
        ("The basic health unit is closed most days", "basic health unit aksar band rehta hai", 2),
    ],
    "Education": [
        ("Teachers are often absent from the government school", "sarkari school mein ustad aksar ghair hazir rehte hain", 1),
        ("The school roof leaks during rain", "barish mein school ki chat tapakti hai", 2),
        ("There are no toilets for girls in the school", "school mein larkiyon ke liye washroom nahi hain", 2),
        ("Students have not received their free textbooks", "talba ko abhi tak muft kitabein nahi mili", 1),
        ("The classrooms are overcrowded with no benches", "classrooms mein bohat bheed hai aur benches nahi hain", 1),
        ("There is no drinking water for students at school", "school mein bachon ke liye peenay ka pani nahi hai", 2),
        ("The school was closed for weeks without any reason", "school bina wajah kai hafton se band hai", 2),
        ("The scholarship payment has been delayed for months", "wazifa kai mahine se nahi mila", 1),
        ("The school building walls are cracked and unsafe", "school ki imarat ki deewarein toot chuki hain aur ghair mehfooz hain", 3),
        ("The school has no science lab or computers", "school mein science lab aur computers nahi hain", 0),
    ],
    "Traffic": [
        ("The traffic signal is not working at the main chowk", "main chowk par traffic signal kharab hai", 1),
        ("There is heavy traffic jam every evening", "har shaam bohat zyada traffic jam hota hai", 1),
        ("Vehicles drive on the wrong side and nobody stops them", "gaariyan ulti taraf se chalti hain koi nahi rokta", 2),
        ("Illegal parking blocks the whole road", "ghair qanooni parking se poori sarak band ho jati hai", 1),
        ("There are frequent accidents at this intersection", "is chowrahe par aye din hadsay hote hain", 3),
        ("No traffic warden is present at the busy crossing", "masroof chowrahe par traffic warden maujood nahi hota", 1),
        ("Overloaded rickshaws and buses are a danger to the public", "zyada sawariyan bithanay wale rickshaw aur bus khatarnak hain", 2),
        ("There is no zebra crossing near the school", "school ke paas zebra crossing nahi hai", 1),
        ("Heavy trucks are allowed in the city during the day", "bhari trucks ko din mein sheher mein dakhil hone diya jata hai", 1),
        ("Motorcycles race on the road and risk lives", "motorcycle sawar tez raftari se chalate hain jaan ka khatra hai", 2),
    ],
    "General": [
        ("The complaint office does not answer any calls", "shikayat daftar koi call receive nahi karta", 1),
        ("Stray dogs are attacking people in our area", "awara kutte logon par hamla kar rahe hain", 2),
        ("Shops have encroached on the footpath", "dukandaron ne footpath par qabza kar liya hai", 0),
        ("The park needs benches and proper lighting", "park mein benches aur roshni ki zaroorat hai", 0),
        ("Loudspeakers create noise all night", "poori raat loudspeaker ka shor hota hai", 1),
        ("The government office staff behaved badly with us", "sarkari daftar ke ahlkaar ne hamare sath bura sulook kiya", 1),
        ("Illegal construction is going on in the residential area", "rehaishi ilaqay mein ghair qanooni tameer ho rahi hai", 1),
        ("The official website has no clear information on services", "official website par khidmaat ki wazeh maloomat nahi hai", 0),
        ("My application has been pending for months with no update", "meri darkhwast kai mahine se zer e iltawa hai koi jawab nahi", 1),
        ("Please add a mobile app to track complaint status", "shikayat ka status dekhne ke liye mobile app banayein", 0),
    ],
}

POS = {
    "Roads / Infrastructure": [("Thank you for repairing the road in our area, it is much better now", "hamare area ki road theek karne ka shukriya ab bohat behtar hai"),
                               ("The new street construction was completed on time", "nayi sarak ki tameer waqt par mukammal hui")],
    "Water Supply": [("Water supply has improved a lot after the new pipeline", "nayi pipeline ke baad pani ki supply bohat behtar ho gayi hai"),
                     ("The water team fixed the leakage quickly, great work", "pani ki team ne leakage jaldi theek ki shabash")],
    "Electricity": [("The new transformer solved our voltage problem, thank you", "naye transformer se voltage ka masla hal ho gaya shukriya"),
                    ("The electricity staff responded quickly to our complaint", "bijli ke amlay ne hamari shikayat par jaldi amal kiya")],
    "Sanitation": [("Garbage is now collected daily, good improvement", "ab kooda rozana uthaya jata hai acha iqdam hai"),
                   ("The sewer was cleaned very fast, appreciated", "gutter ki safai bohat jaldi hui shukriya")],
    "Health": [("The hospital staff was very helpful and polite", "hospital ka staff bohat madadgar aur shaista tha"),
               ("Doctors treated my mother well and the hospital was clean", "doctors ne meri walida ka acha ilaj kiya aur hospital saaf tha")],
    "Education": [("The teachers in our school are very dedicated, thank you", "hamare school ke ustad bohat mehnati hain shukriya"),
                  ("Free books were delivered on time this year", "is saal muft kitabein waqt par mil gayin")],
    "Traffic": [("The new signal has reduced the traffic jam, good job", "naye signal se traffic jam kam ho gaya acha kaam hai"),
                ("Traffic wardens are doing a great job at the chowk", "chowk par traffic wardens acha kaam kar rahe hain")],
    "General": [("The complaint helpline answered quickly and solved my issue", "shikayat helpline ne jaldi jawab diya aur masla hal hua"),
                ("The park has been renovated nicely, thanks to the team", "park ki acchi marammat hui team ka shukriya")],
}

NEU = {
    "Roads / Infrastructure": [("How can I report a damaged road to the municipal office?", "kharab road ki shikayat municipal office mein kaise karun?"),
                               ("When will the road construction in sector be completed?", "sector mein sarak ki tameer kab mukammal hogi?")],
    "Water Supply": [("What are the water supply timings for our block?", "hamare block mein pani ki supply ke auqat kya hain?"),
                     ("Where can I apply for a new water connection?", "naye pani connection ke liye kahan apply karun?")],
    "Electricity": [("How do I apply for a new electricity meter?", "naye bijli meter ke liye kaise apply karun?"),
                    ("What is the schedule of load shedding this week?", "is hafte load shedding ka schedule kya hai?")],
    "Sanitation": [("What time does the garbage truck visit our street?", "kooda uthane wali gaari kis waqt aati hai?"),
                   ("Which office handles sewer line complaints?", "gutter ki shikayat kaun sa daftar sunta hai?")],
    "Health": [("What are the OPD timings of the government hospital?", "sarkari hospital ki OPD ke auqat kya hain?"),
               ("Is the vaccination camp open this Sunday?", "kya is itwaar ko vaccination camp khula hoga?")],
    "Education": [("When does admission start in government schools?", "sarkari schools mein dakhlay kab shuru hotay hain?"),
                  ("How can I apply for the school scholarship?", "school wazifay ke liye kaise apply karun?")],
    "Traffic": [("Which route is open for traffic during the construction?", "tameeri kaam ke dauran kaun sa rasta khula hai?"),
                ("Where can I check my traffic challan online?", "apna traffic challan online kahan dekhun?")],
    "General": [("What is the phone number of the citizen helpline?", "citizen helpline ka number kya hai?"),
                ("How do I check the status of my application?", "apni darkhwast ka status kaise check karun?")],
}

OPENERS_EN = ["", "", "", "Sir, ", "Dear officer, ", "Respected sir, ", "Hello, ", "Complaint: ", "I want to report that "]
OPENERS_UR = ["", "", "", "Sir, ", "Janab, ", "Assalam o alaikum, ", "Meherbani karke note karein, ", "Shikayat: "]
# (phrase_en, phrase_ur, severity bump)
TIME = [("", "", 0), ("", "", 0), ("", "", 0),
        (" since yesterday", " kal se", 0), (" for two days", " do din se", 0),
        (" for three days", " teen din se", 1), (" for a week", " ek hafte se", 1),
        (" for many weeks", " kai hafton se", 1), (" for months", " kai mahine se", 1),
        (" every day", " rozana", 0)]
CLOSERS = [("", "", 0), ("", "", 0), ("", "", 0), (" Please look into it.", " Meherbani karke dekhein.", 0),
           (" Kindly take action.", " Barae meherbani karwai karein.", 0),
           (" Please fix it urgently.", " Baraye karam jaldi theek karwayein.", 1),
           (" This is an emergency, act immediately!", " Yeh emergency hai fori karwai karein!", 1),
           (" It is a serious problem for everyone.", " Yeh sab ke liye bara masla hai.", 0)]
LOCS_EN = ["", "", " in our area", " in our locality", " near the main market", " in our neighbourhood", " in Block C", " in our colony"]
LOCS_UR = ["", "", " hamare area mein", " hamare muhalle mein", " bazar ke qareeb", " hamari colony mein"]

NOISE_PHONE = [" Call me 0300-1234567.", " Contact: 0321 7654321", " my number is 03451112223"]
NOISE_URL = [" see http://example.com/photo123", " https://bit.ly/abc123 pic attached", " www.example.org/complaint"]
NOISE_EMOJI = [" 😡", " 🙏", " 😞", " 👎"]
TYPOS = {"electricity": "electrcity", "garbage": "garbge", "hospital": "hosptial", "water": "watter",
         "school": "scool", "traffic": "trafic", "road": "rode", "please": "plz", "urgently": "urgentlyy"}

POS_PRE_EN = ["", "", "Thank you, ", "Good news: ", "Just wanted to say, ", "Appreciation: ", "Great work, "]
POS_SUF_EN = ["", "", " Keep it up.", " Well done team.", " Very satisfied.", " Hope it continues."]
POS_PRE_UR = ["", "", "Shukriya, ", "Achi khabar: ", "Bas yeh kehna tha, ", "Tareef: "]
POS_SUF_UR = ["", "", " Aisay hi karte rahein.", " Shabash team.", " Hum bohat khush hain.", " Umeed hai yeh jari rahega."]
NEU_PRE_EN = ["", "", "Hello, ", "Please tell me, ", "Sir, ", "Quick question: "]
NEU_PRE_UR = ["", "", "Assalam o alaikum, ", "Meherbani karke bataein, ", "Janab, ", "Ek sawal hai: "]

PRIORITY_NAMES = ["Low", "Medium", "High", "Urgent"]


def priority_from(sev, bump_time, bump_close):
    total = sev + bump_time + bump_close
    if total >= 3 and sev >= 2:
        return 3
    return max(0, min(total, 2)) if sev < 3 else 3


def add_noise(text):
    r = random.random()
    if r < 0.04:
        text += random.choice(NOISE_PHONE)
    elif r < 0.07:
        text += random.choice(NOISE_URL)
    elif r < 0.09:
        text += random.choice(NOISE_EMOJI)
    r = random.random()
    if r < 0.05:
        for k, v in TYPOS.items():
            if k in text.lower():
                text = text.replace(k, v).replace(k.capitalize(), v.capitalize())
                break
    r = random.random()
    if r < 0.04:
        text = text.upper()
    elif r < 0.10:
        text = text.lower()
    if random.random() < 0.05:
        text = "  " + text.replace(" ", "  ", 1) + "  "
    return text


def sentence(en, ur, lang, opener_pool=True):
    """Compose one negative sample; returns text, time_bump, close_bump."""
    t_en, t_ur, tb = random.choice(TIME)
    c_en, c_ur, cb = random.choice(CLOSERS)
    l_en, l_ur = random.choice(LOCS_EN), random.choice(LOCS_UR)
    if " in our" in en or " near the" in en or " in the" in en:
        l_en = ""
    if lang == "english":
        o = random.choice(OPENERS_EN)
        body = en if o == "" or o.endswith("that ") is False else en[0].lower() + en[1:]
        if o.endswith("that "):
            body = en[0].lower() + en[1:]
        text = f"{o}{body}{l_en}{t_en}.{c_en}"
    elif lang == "roman_urdu":
        o = random.choice(OPENERS_UR)
        text = f"{o}{ur}{t_ur}.{c_ur}"
        text = text.replace(" .", ".")
    else:  # mixed: English issue + Roman Urdu closer/time, or vice versa
        if random.random() < 0.5:
            text = f"{en}{l_en}.{c_ur}"
        else:
            text = f"{ur}{t_en}.{c_en}"
        # time/closer bump only valid when they actually appear
        tb = tb if (t_en and text.endswith(c_en) is not None and t_en in text) else 0
    return text.strip(), tb, cb


def main(n_neg=900, n_pos=260, n_neu=260):
    rows = []
    cats = list(NEG.keys())
    langs = ["english"] * 45 + ["roman_urdu"] * 35 + ["mixed"] * 20

    for i in range(n_neg):
        cat = cats[i % len(cats)]
        en, ur, sev = random.choice(NEG[cat])
        lang = random.choice(langs)
        text, tb, cb = sentence(en, ur, lang)
        # cb only counts if closer shows up in text
        if cb and not any(k in text for k in ["urgent", "emergency", "jaldi", "fori"]):
            cb = 0
        pr = priority_from(sev, tb, cb)
        rows.append((text, cat, "Negative", PRIORITY_NAMES[pr], lang))

    # Multi-issue (ambiguous) complaints: labelled by the PRIMARY problem.
    # These are useful for error analysis (e.g. road + water).
    multi = [
        ("The road is broken and rainwater is standing on it for days", "Roads / Infrastructure", "Negative", "Medium"),
        ("Sewage water is flooding the road near the school", "Sanitation", "Negative", "High"),
        ("Hospital has no electricity and patients are suffering", "Health", "Negative", "Urgent"),
        ("School has no water supply and the toilets are dirty", "Education", "Negative", "High"),
        ("Traffic signal is off because of the power cut", "Traffic", "Negative", "Medium"),
        ("Garbage on the road is blocking traffic", "Sanitation", "Negative", "Medium"),
        ("road pe pani jama hai aur gutter bhi ubal raha hai", "Sanitation", "Negative", "High"),
        ("hospital ke bahar sarak toot chuki hai ambulance nahi guzar sakti", "Roads / Infrastructure", "Negative", "Urgent"),
        ("bijli nahi hai to tube well band hai aur pani nahi aa raha", "Electricity", "Negative", "High"),
        ("school ki road par traffic jam rehta hai", "Traffic", "Negative", "Medium"),
        ("Doctor was helpful but the waiting time was too long", "Health", "Negative", "Medium"),
        ("staff acha tha lekin hospital mein dawai nahi thi", "Health", "Negative", "High"),
        ("Water tanker blocks the road every morning", "Traffic", "Negative", "Low"),
        ("The park is dirty and the street lights are off", "General", "Negative", "Medium"),
        ("pani ka bill aur bijli ka bill dono galat aaye hain", "General", "Negative", "Low"),
        ("No electricity, no water, and nobody answers the complaint line", "General", "Negative", "High"),
    ]
    for t, c, s, p in multi:
        for _ in range(2):  # repeat; noise will differentiate copies
            rows.append((t, c, s, p, "multi_issue"))

    for i in range(n_pos):
        cat = cats[i % len(cats)]
        en, ur = random.choice(POS[cat])
        lang = random.choice(["english", "roman_urdu", "mixed"])
        if lang == "english":
            text = random.choice(POS_PRE_EN) + en.rstrip(".") + "." + random.choice(POS_SUF_EN)
        elif lang == "roman_urdu":
            text = random.choice(POS_PRE_UR) + ur.rstrip(".") + "." + random.choice(POS_SUF_UR)
        else:
            text = en.split(",")[0] + ". " + random.choice(POS_SUF_UR).strip() + " Shukriya."
        rows.append((text.strip(), cat, "Positive", "Low", lang))

    for i in range(n_neu):
        cat = cats[i % len(cats)]
        en, ur = random.choice(NEU[cat])
        lang = random.choice(["english", "roman_urdu"])
        base = (random.choice(NEU_PRE_EN) + en) if lang == "english" else (random.choice(NEU_PRE_UR) + ur)
        rows.append((base, cat, "Neutral", random.choice(["Low", "Low", "Medium"]), lang))

    out = []
    for text, cat, sen, pri, lang in rows:
        out.append({"text": add_noise(text), "category": cat, "sentiment": sen, "priority": pri,
                    "language": lang, "label_source": "synthetic_rule"})

    df = pd.DataFrame(out)
    # a few missing values + exact duplicates (to be removed in cleaning)
    for idx in random.sample(range(len(df)), 6):
        df.loc[idx, "text"] = random.choice(["", None, "   "])
    dups = df.sample(60, random_state=1)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    path = ROOT / "data" / "raw" / "feedback_raw.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} raw rows -> {path}")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
