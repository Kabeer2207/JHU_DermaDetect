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
            "Use gentle skin-care products",
            "Apply broad-spectrum sunscreen daily"
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
            "Avoid harsh soaps and known irritants",
            "Wear soft, breathable fabrics",
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
            "Avoid excessive scrubbing",
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
            "Moisturize skin regularly",
            "Avoid known triggers such as stress or skin injury",
            "Follow a personalized treatment plan if prescribed"
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
            "Protect skin from sun exposure",
            "Use cosmetic camouflage if desired",
            "Seek emotional and psychological support if needed"
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
            "Monitor skin for any future changes"
        ]
    }
}
