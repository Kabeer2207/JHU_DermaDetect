"""
diagnosis_data.py

This file contains medical metadata for each skin condition
supported by the DermaDetect model.

IMPORTANT:
- This information is for educational and informational purposes only.
- It is NOT a substitute for professional medical advice.
"""

DIAGNOSIS_DATA = {
    "Acne_Rosacea": {
        "name": "Acne Rosacea",
        "severity": "Medium Severity",
        "description": (
            "A chronic skin condition that primarily affects the face, "
            "causing redness, visible blood vessels, and sometimes acne-like bumps."
        ),
        "symptoms": [
            "Persistent facial redness",
            "Visible blood vessels",
            "Swollen red bumps",
            "Burning or stinging sensation"
        ],
        "recommendations": [
            "Consult a dermatologist for diagnosis confirmation",
            "Avoid known triggers such as spicy food, alcohol, and extreme temperatures",
            "Use gentle, fragrance-free skin-care products",
            "Apply broad-spectrum sunscreen daily",
            "Limit exposure to extreme heat and sunlight",
            "Avoid abrasive exfoliation or harsh treatments",
            "Cleanse the face with lukewarm water only",
            "Monitor flare-ups and note potential triggers"
        ]
    },

    "Eczema": {
        "name": "Eczema",
        "severity": "Medium Severity",
        "description": (
            "A condition that causes the skin to become dry, itchy, inflamed, "
            "and sometimes cracked or scaly."
        ),
        "symptoms": [
            "Dry or sensitive skin",
            "Intense itching",
            "Red or inflamed patches",
            "Thickened or scaly skin"
        ],
        "recommendations": [
            "Moisturize skin regularly using fragrance-free products",
            "Avoid harsh soaps, detergents, and known irritants",
            "Wear soft, breathable fabrics such as cotton",
            "Limit hot showers and prolonged water exposure",
            "Apply moisturizers immediately after bathing",
            "Identify and avoid personal eczema triggers",
            "Use a humidifier in dry environments",
            "Seek medical advice for persistent or severe symptoms"
        ]
    },

    "Keratosis": {
        "name": "Keratosis",
        "severity": "Low Severity",
        "description": (
            "A common skin condition characterized by rough, dry patches "
            "or small bumps, often on the arms or thighs."
        ),
        "symptoms": [
            "Rough or dry skin texture",
            "Small raised bumps",
            "Skin discoloration in affected areas"
        ],
        "recommendations": [
            "Maintain regular skin hydration",
            "Use gentle exfoliation if recommended",
            "Avoid excessive scrubbing or friction",
            "Apply moisturizers containing mild exfoliating agents",
            "Limit long, hot showers that dry the skin",
            "Wear loose-fitting clothing to reduce irritation",
            "Be consistent with daily skin care routines",
            "Consult a dermatologist for persistent cases"
        ]
    },

    "Milia": {
        "name": "Milia",
        "severity": "Low Severity",
        "description": (
            "Small, white cysts that appear on the skin when keratin becomes "
            "trapped beneath the surface."
        ),
        "symptoms": [
            "Tiny white or yellowish bumps",
            "Firm, raised cysts",
            "Typically appear around eyes or cheeks"
        ],
        "recommendations": [
            "Avoid squeezing or picking at the bumps",
            "Maintain proper skin hygiene",
            "Use non-comedogenic skincare products",
            "Avoid heavy or oily cosmetics",
            "Gently cleanse the affected area daily",
            "Limit excessive exfoliation",
            "Protect skin from unnecessary irritation",
            "Seek professional removal if desired"
        ]
    },

    "Psoriasis": {
        "name": "Psoriasis",
        "severity": "High Severity",
        "description": (
            "A chronic autoimmune condition that causes rapid skin cell buildup, "
            "leading to thick, scaly patches on the skin."
        ),
        "symptoms": [
            "Thick, red patches with silvery scales",
            "Dry or cracked skin",
            "Itching or soreness",
            "Changes in nail texture"
        ],
        "recommendations": [
            "Consult a dermatologist for long-term management",
            "Moisturize skin regularly to reduce dryness",
            "Avoid known triggers such as stress or skin injury",
            "Follow a personalized treatment plan if prescribed",
            "Limit alcohol consumption and smoking",
            "Use gentle, non-irritating skincare products",
            "Protect skin from extreme weather conditions",
            "Monitor flare-ups and symptom patterns"
        ]
    },

    "Ringworm": {
        "name": "Ringworm",
        "severity": "Medium Severity",
        "description": (
            "A fungal skin infection that appears as a circular, itchy rash "
            "with clearer skin in the center."
        ),
        "symptoms": [
            "Circular red or scaly rash",
            "Itching or irritation",
            "Raised borders around the affected area"
        ],
        "recommendations": [
            "Keep the affected area clean and dry",
            "Avoid sharing personal items such as towels",
            "Wash hands thoroughly after touching the area",
            "Change clothing and bedding regularly",
            "Avoid tight or non-breathable clothing",
            "Do not scratch the affected skin",
            "Consult a healthcare professional for treatment guidance",
            "Complete the full course of recommended care"
        ]
    },

    "Vitiligo": {
        "name": "Vitiligo",
        "severity": "Medium Severity",
        "description": (
            "A condition in which pigment-producing cells are lost, "
            "resulting in white patches on the skin."
        ),
        "symptoms": [
            "Loss of skin color in patches",
            "Premature whitening of hair",
            "Changes in pigmentation of mucous membranes"
        ],
        "recommendations": [
            "Consult a dermatologist for evaluation",
            "Protect skin from sun exposure using sunscreen",
            "Use cosmetic camouflage if desired",
            "Avoid skin trauma or unnecessary friction",
            "Maintain consistent skincare routines",
            "Monitor changes in skin pigmentation",
            "Seek emotional and psychological support if needed",
            "Stay informed about management options"
        ]
    },

    "Normal_Skin": {
        "name": "Normal Skin",
        "severity": "Healthy Skin",
        "description": (
            "The skin appears healthy with no visible signs of disease or abnormality."
        ),
        "symptoms": [
            "Even skin tone",
            "No visible irritation or lesions",
            "Healthy skin texture"
        ],
        "recommendations": [
            "Maintain regular skincare and hygiene",
            "Use sunscreen to protect against UV damage",
            "Stay hydrated and maintain a balanced diet",
            "Cleanse skin gently on a daily basis",
            "Moisturize to preserve skin barrier health",
            "Avoid excessive use of harsh products",
            "Monitor skin for any future changes",
            "Schedule routine skin check-ups if needed"
        ]
    }
}
