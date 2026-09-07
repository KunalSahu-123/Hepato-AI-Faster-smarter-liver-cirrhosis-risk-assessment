"""
recommendations.py
-------------------
Provides doctor and hospital recommendations based on liver cirrhosis risk
assessment results. Curated list of top Indian hepatologists, liver transplant
surgeons, and leading hospitals for demonstration purposes.

Location-aware sorting: specialists can be sorted by proximity to the user
using either browser geolocation or IP-based city detection.
"""

import math

SPECIALISTS = [
    # --- Critical / Transplant-level specialists ---
    {
        "name": "Dr. A. S. Soin",
        "specialisation": "Liver Transplant & Hepatobiliary Surgery",
        "hospital": "Medanta - The Medicity",
        "city": "Gurugram, Haryana",
        "address": "CH Baktawar Singh Rd, Sector 38, Gurugram 122001",
        "phone": "+91-124-4141414",
        "email": "info@medanta.org",
        "website": "https://www.medanta.org",
        "maps": "https://maps.google.com/?q=Medanta+The+Medicity+Gurugram",
        "rating": 4.9,
        "experience": "30+ years",
        "tier": "critical",
        "lat": 28.4395,
        "lng": 77.0266,
    },
    {
        "name": "Dr. Subhash Gupta",
        "specialisation": "Liver Transplant & Hepatobiliary Surgery",
        "hospital": "Max Super Speciality Hospital, Saket",
        "city": "New Delhi",
        "address": "1, 2, Press Enclave Rd, Saket, New Delhi 110017",
        "phone": "+91-11-26515050",
        "email": "info.saket@maxhealthcare.com",
        "website": "https://www.maxhealthcare.in",
        "maps": "https://maps.google.com/?q=Max+Super+Speciality+Hospital+Saket+Delhi",
        "rating": 4.9,
        "experience": "28+ years",
        "tier": "critical",
        "lat": 28.5274,
        "lng": 77.2130,
    },
    {
        "name": "Dr. Vivek Vij",
        "specialisation": "Liver Transplant & HPB Surgery",
        "hospital": "Fortis Memorial Research Institute",
        "city": "Gurugram, Haryana",
        "address": "Sector 44, Opposite HUDA City Centre, Gurugram 122002",
        "phone": "+91-124-4962200",
        "email": "enquiries@fortishealthcare.com",
        "website": "https://www.fortishealthcare.com",
        "maps": "https://maps.google.com/?q=Fortis+Memorial+Research+Institute+Gurugram",
        "rating": 4.8,
        "experience": "22+ years",
        "tier": "critical",
        "lat": 28.4595,
        "lng": 77.0723,
    },
    {
        "name": "Dr. Sanjay Singh Negi",
        "specialisation": "Liver Transplant & GI Surgery",
        "hospital": "ILBS (Institute of Liver and Biliary Sciences)",
        "city": "New Delhi",
        "address": "D-1, Vasant Kunj, New Delhi 110070",
        "phone": "+91-11-46300000",
        "email": "info@ilbs.in",
        "website": "https://www.ilbs.in",
        "maps": "https://maps.google.com/?q=ILBS+Institute+of+Liver+Biliary+Sciences+Delhi",
        "rating": 4.8,
        "experience": "25+ years",
        "tier": "critical",
        "lat": 28.5189,
        "lng": 77.1540,
    },
    {
        "name": "Dr. Mohamed Rela",
        "specialisation": "Liver Transplant & HPB Surgery",
        "hospital": "Dr. Rela Institute & Medical Centre",
        "city": "Chennai, Tamil Nadu",
        "address": "No. 7, CLC Works Rd, Chromepet, Chennai 600044",
        "phone": "+91-44-66050605",
        "email": "info@relainstitute.com",
        "website": "https://www.relainstitute.com",
        "maps": "https://maps.google.com/?q=Dr+Rela+Institute+Medical+Centre+Chennai",
        "rating": 4.9,
        "experience": "35+ years",
        "tier": "critical",
        "lat": 12.9516,
        "lng": 80.1462,
    },
    # --- High-risk / Hepatology specialists ---
    {
        "name": "Dr. Sanjiv Saigal",
        "specialisation": "Hepatology & Gastroenterology",
        "hospital": "Sir Ganga Ram Hospital",
        "city": "New Delhi",
        "address": "Rajinder Nagar, New Delhi 110060",
        "phone": "+91-11-25750000",
        "email": "info@sgrh.com",
        "website": "https://www.sgrh.com",
        "maps": "https://maps.google.com/?q=Sir+Ganga+Ram+Hospital+Delhi",
        "rating": 4.7,
        "experience": "25+ years",
        "tier": "high",
        "lat": 28.6399,
        "lng": 77.1850,
    },
    {
        "name": "Dr. T. G. Balachandar",
        "specialisation": "Gastroenterology & Hepatology",
        "hospital": "Apollo Hospitals, Greams Road",
        "city": "Chennai, Tamil Nadu",
        "address": "21, Greams Ln, Off Greams Rd, Chennai 600006",
        "phone": "+91-44-28290200",
        "email": "enquiry@apollohospitals.com",
        "website": "https://www.apollohospitals.com",
        "maps": "https://maps.google.com/?q=Apollo+Hospital+Greams+Road+Chennai",
        "rating": 4.8,
        "experience": "20+ years",
        "tier": "high",
        "lat": 13.0604,
        "lng": 80.2496,
    },
    {
        "name": "Dr. Naimish Mehta",
        "specialisation": "Hepatology & Liver Diseases",
        "hospital": "Global Hospitals, Parel",
        "city": "Mumbai, Maharashtra",
        "address": "35, Dr E Moses Rd, Worli, Mumbai 400018",
        "phone": "+91-22-67670101",
        "email": "info.mumbai@globalhospitalsindia.com",
        "website": "https://www.globalhospitalsindia.com",
        "maps": "https://maps.google.com/?q=Global+Hospitals+Parel+Mumbai",
        "rating": 4.7,
        "experience": "18+ years",
        "tier": "high",
        "lat": 19.0033,
        "lng": 72.8420,
    },
    {
        "name": "Dr. Manav Wadhawan",
        "specialisation": "Gastroenterology & Hepatology",
        "hospital": "BLK-Max Super Speciality Hospital",
        "city": "New Delhi",
        "address": "Pusa Road, Rajinder Nagar, New Delhi 110005",
        "phone": "+91-11-30403040",
        "email": "info@blkmaxhospital.com",
        "website": "https://www.blkmaxhospital.com",
        "maps": "https://maps.google.com/?q=BLK+Max+Hospital+Delhi",
        "rating": 4.6,
        "experience": "22+ years",
        "tier": "high",
        "lat": 28.6436,
        "lng": 77.1810,
    },
    {
        "name": "Dr. Anil Arora",
        "specialisation": "Gastroenterology & Hepatology",
        "hospital": "Sir Ganga Ram Hospital",
        "city": "New Delhi",
        "address": "Rajinder Nagar, New Delhi 110060",
        "phone": "+91-11-25750000",
        "email": "info@sgrh.com",
        "website": "https://www.sgrh.com",
        "maps": "https://maps.google.com/?q=Sir+Ganga+Ram+Hospital+Delhi",
        "rating": 4.7,
        "experience": "30+ years",
        "tier": "high",
        "lat": 28.6399,
        "lng": 77.1850,
    },
    {
        "name": "Dr. Shiv Kumar Sarin",
        "specialisation": "Hepatology & Liver Sciences",
        "hospital": "ILBS (Institute of Liver and Biliary Sciences)",
        "city": "New Delhi",
        "address": "D-1, Vasant Kunj, New Delhi 110070",
        "phone": "+91-11-46300000",
        "email": "info@ilbs.in",
        "website": "https://www.ilbs.in",
        "maps": "https://maps.google.com/?q=ILBS+Institute+of+Liver+Biliary+Sciences+Delhi",
        "rating": 4.9,
        "experience": "35+ years",
        "tier": "high",
        "lat": 28.5189,
        "lng": 77.1540,
    },
    # --- Moderate / Routine specialists ---
    {
        "name": "Dr. Pankaj Puri",
        "specialisation": "Gastroenterology & Hepatology",
        "hospital": "Fortis Escorts Heart Institute",
        "city": "New Delhi",
        "address": "Okhla Road, New Delhi 110025",
        "phone": "+91-11-47135000",
        "email": "enquiries@fortishealthcare.com",
        "website": "https://www.fortishealthcare.com",
        "maps": "https://maps.google.com/?q=Fortis+Escorts+Heart+Institute+Delhi",
        "rating": 4.6,
        "experience": "20+ years",
        "tier": "moderate",
        "lat": 28.5571,
        "lng": 77.2720,
    },
    {
        "name": "Dr. Akash Shukla",
        "specialisation": "Hepatology & Gastroenterology",
        "hospital": "Sir H. N. Reliance Foundation Hospital",
        "city": "Mumbai, Maharashtra",
        "address": "Raja Rammohan Roy Rd, Prarthana Samaj, Girgaon, Mumbai 400004",
        "phone": "+91-22-61305000",
        "email": "info@rfhospital.org",
        "website": "https://www.rfhospital.org",
        "maps": "https://maps.google.com/?q=Reliance+Foundation+Hospital+Mumbai",
        "rating": 4.7,
        "experience": "18+ years",
        "tier": "moderate",
        "lat": 18.9560,
        "lng": 72.8132,
    },
    {
        "name": "Dr. Dharmesh Kapoor",
        "specialisation": "Hepatology & Gastroenterology",
        "hospital": "Global Hospitals, Lakdi-ka-Pul",
        "city": "Hyderabad, Telangana",
        "address": "6-1-1070/1 to 4, Lakdi-ka-Pul, Hyderabad 500004",
        "phone": "+91-40-30244444",
        "email": "info.hyderabad@globalhospitalsindia.com",
        "website": "https://www.globalhospitalsindia.com",
        "maps": "https://maps.google.com/?q=Global+Hospitals+Hyderabad",
        "rating": 4.6,
        "experience": "22+ years",
        "tier": "moderate",
        "lat": 17.4006,
        "lng": 78.4683,
    },
    {
        "name": "Dr. Deepak Amarapurkar",
        "specialisation": "Gastroenterology & Hepatology",
        "hospital": "Bombay Hospital & Medical Research Centre",
        "city": "Mumbai, Maharashtra",
        "address": "12, New Marine Lines, Mumbai 400020",
        "phone": "+91-22-22067676",
        "email": "info@bombayhospital.com",
        "website": "https://www.bombayhospital.com",
        "maps": "https://maps.google.com/?q=Bombay+Hospital+Mumbai",
        "rating": 4.5,
        "experience": "30+ years",
        "tier": "moderate",
        "lat": 18.9432,
        "lng": 72.8266,
    },
    {
        "name": "Dr. Cyriac Abby Philips",
        "specialisation": "Clinical Hepatology & Liver ICU",
        "hospital": "The Liver Institute, CLMC",
        "city": "Ernakulam, Kerala",
        "address": "Rajagiri Rd, Chunangamvely, Aluva, Ernakulam 683112",
        "phone": "+91-484-2905000",
        "email": "info@rajagirihospital.com",
        "website": "https://www.rajagirihospital.com",
        "maps": "https://maps.google.com/?q=Rajagiri+Hospital+Ernakulam+Kerala",
        "rating": 4.7,
        "experience": "15+ years",
        "tier": "moderate",
        "lat": 10.0610,
        "lng": 76.3520,
    },
]


def _haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two points in kilometres."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sort_by_distance(specialists, lat, lng):
    """Return specialists sorted by distance from (lat, lng), with distance_km added."""
    enriched = []
    for doc in specialists:
        dist = _haversine_km(lat, lng, doc["lat"], doc["lng"])
        enriched.append({**doc, "distance_km": round(dist)})
    enriched.sort(key=lambda d: d["distance_km"])
    return enriched


def get_recommendations(risk_level, result_label, user_lat=None, user_lng=None):
    """Return doctor/hospital recommendations based on assessment outcome.

    Parameters
    ----------
    risk_level : str
        "Low", "Moderate", "High", or "Critical" risk band.
    result_label : str
        "High Risk" or "Low Risk" prediction.

    Returns
    -------
    dict with keys:
        urgency     - "immediate", "soon", or "routine"
        message     - plain-language recommendation
        specialists - list of specialist dicts (filtered by tier)
        actions     - list of recommended next-step strings
    """
    if result_label == "High Risk" and risk_level in ("Critical", "High"):
        urgency = "immediate"
        message = (
            "Your assessment indicates elevated liver cirrhosis risk. "
            "We strongly recommend consulting a liver transplant specialist "
            "or hepatologist at the earliest. The specialists listed below "
            "are among India's top-rated liver experts."
        )
        actions = [
            "Schedule an urgent appointment with a hepatologist or liver transplant specialist",
            "Get a comprehensive liver function test (LFT) and complete blood count (CBC)",
            "Request abdominal ultrasound with Doppler and FibroScan (transient elastography)",
            "Get tested for Hepatitis B & C if not done recently",
            "Bring this report, lab values, and any imaging reports to the consultation",
            "Avoid alcohol completely and follow a low-sodium diet until further advice",
        ]
        specialists = [s for s in SPECIALISTS if s["tier"] in ("critical", "high")]
    elif result_label == "High Risk" or risk_level == "Moderate":
        urgency = "soon"
        message = (
            "Your results show moderate risk indicators. A consultation "
            "with a gastroenterologist or hepatologist within the next 2-4 "
            "weeks is recommended. The specialists below can help with "
            "further evaluation and monitoring."
        )
        actions = [
            "Book a consultation with a gastroenterologist or hepatologist within 2-4 weeks",
            "Repeat liver function tests (LFT) after 4-6 weeks",
            "Maintain a liver-friendly diet: low sodium, high protein, avoid processed foods",
            "Avoid alcohol completely",
            "Monitor for symptoms like jaundice, fatigue, or abdominal swelling",
            "Share this report with your primary care physician",
        ]
        specialists = [s for s in SPECIALISTS if s["tier"] in ("high", "moderate")]
    else:
        urgency = "routine"
        message = (
            "Your assessment shows low risk. Continue routine health "
            "check-ups and maintain a healthy lifestyle. The specialists "
            "below are available for preventive consultations if needed."
        )
        actions = [
            "Continue routine health check-ups annually",
            "Maintain a balanced diet rich in fruits, vegetables, and lean protein",
            "Limit alcohol consumption or avoid it entirely",
            "Stay up to date on Hepatitis A and B vaccinations",
            "Exercise regularly and maintain a healthy weight",
            "Report any new symptoms (fatigue, jaundice, swelling) to your doctor",
        ]
        specialists = [s for s in SPECIALISTS if s["tier"] == "moderate"]

    if user_lat is not None and user_lng is not None:
        specialists = sort_by_distance(specialists, user_lat, user_lng)

    return {
        "urgency": urgency,
        "message": message,
        "specialists": specialists,
        "actions": actions,
    }
