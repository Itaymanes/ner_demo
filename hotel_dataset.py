"""
Synthetic hotel dataset for GLiNER evaluation
"""

HOTEL_DATASET = [
    {
        "text": """
About this property
Prime City Centre Location: Maccani Black Luxury Suites in Belgrade offers a convenient location with Republic Square just a 6-minute walk away. The National Assembly lies 400 metres nearby, while Belgrade Nikola Tesla Airport is 12 km from the property.
Comfortable Accommodations: Guests enjoy free WiFi, air-conditioning, private bathrooms with walk-in showers, and city views. Additional amenities include bathrobes, minibars, and soundproofed rooms.

Exceptional Services: The guest house provides a paid shuttle service, lift, housekeeping, family rooms, full-day security, express check-in and check-out, and luggage storage. On-site private parking is available for a fee.

Nearby Attractions: Nearby attractions include Tašmajdan Stadium less than 1 km away, Belgrade Arena and Belgrade Fair 4 km from the property, and an ice-skating rink and boating in the surroundings.

Couples particularly like the location — they rated it 9.6 for a two-person trip.""",
        "entities": [
            {"text": "Maccani Black Luxury Suites", "label": "hotel", "start": 49, "end": 76}
        ]
    },
    {
        "text": "Oceanview Resort & Spa sits on pristine Malibu Beach, California, offering breathtaking Pacific Ocean views. Our 150 spacious suites feature private balconies, marble bathrooms, and complimentary Wi-Fi. The resort includes 3 swimming pools, tennis court, and the renowned Blue Water Grill serving fresh seafood. Rates range from $280-$450 depending on the season.",
        "entities": [
            {"text": "Oceanview Resort & Spa", "label": "hotel", "start": 0, "end": 22}
        ]
    },
    {
        "text": """
About this property
Comfortable Accommodation: The Residence 59 in Belgrade offers a 4-star apartment with a garden and terrace. Guests enjoy free WiFi, air-conditioning, and private bathrooms equipped with modern amenities.

Convenient Facilities: The property provides private check-in and check-out services, a paid shuttle, lift, housekeeping, family rooms, express services, and luggage storage. Paid on-site private parking is available.

Prime Location: Located 8 km from Belgrade Nikola Tesla Airport, the apartment is near Belgrade Arena (6 km), Republic Square (8 km), and Belgrade Train Station (9 km). Guests appreciate the room cleanliness, comfort, and attentive staff.

Couples particularly like the location — they rated it 9.1 for a two-person trip.        
        """,
        "entities": [
            {"text": "The Residence 59", "label": "hotel", "start": 48, "end": 64}
        ]
    },
    {
        "text": """
About this property
Comfortable Accommodations: Hotel Forever in Belgrade offers 4-star comfort with air-conditioning, private bathrooms, and modern amenities. Each room includes a tea and coffee maker, refrigerator, work desk, and free WiFi.

Outdoor Spaces: Guests can relax in the garden or on the terrace, enjoying the serene surroundings. The hotel features an outdoor seating area and bicycle parking for leisure activities.

Convenient Services: Private check-in and check-out, a 24-hour front desk, concierge, and housekeeping services ensure a comfortable stay. Additional amenities include a paid airport shuttle, luggage storage, and paid parking.

Prime Location: Located 14 km from Belgrade Nikola Tesla Airport, Hotel Forever is a short walk from Republic Square and close to attractions such as the National Assembly and Tašmajdan Stadium. Nearby activities include ice-skating and boating.

        """,
        "entities": [
            {"text": "Hotel Forever", "label": "hotel", "start": 49, "end": 62},
            {"text": "Hotel Forever", "label": "hotel", "start": 727, "end": 740}
        ]
    },
]

# Entity labels used in the dataset
HOTEL_LABELS = ["hotel"]