from typing import Dict, List, Any

from tiktoken import encoding_for_model


min_needs_count: int = 1
max_needs_count: int = 3
enc = encoding_for_model("gpt-4")

needs_dict: Dict[str, List[Dict[str, str]]] = {
    "Survival Needs": [
        {"Information Access": "Obtain basic living information such as weather, traffic, news, etc."}, 
        {"Shopping and Consumption": "Purchase daily necessities, clothing, electronic products, etc."},
        {"Service Appointments": "Order takeout, call for transportation, schedule repairs, etc."}
    ],
    "Safety Needs": [
        {"Privacy Protection": "Protect personal information and data security."},
        {"Financial Security": "Security of online banking and investment platforms."},
        {"Health Support": "Online medical consultation and health monitoring services."},
        {"Knowledge Learning": "Obtain reliable information and skills to cope with career and life uncertainties."}
    ],
    "Social Needs": [
        {"Social Interaction": "Connect with others through social media and instant messaging software."},
        {"Expression and Sharing": "Share life, emotions, or opinions through moments, blogs, videos."},
        {"Group Belonging": "Join online communities, interest groups, forums, and interact with like-minded people."}
    ],
    "Esteem Needs": [
        {"Recognition and Achievement": "Gain likes, comments, followers through high-quality content creation (such as short videos, live broadcasts)."},
        {"Personal Brand Building": "Showcase professional capabilities through professional networks (such as LinkedIn) or blogs."},
        {"Authoritative Information Release": "Become an opinion leader or authoritative voice in a specific field."}
    ],
    "Self-Actualization Needs": [
        {"Learning and Growth": "Continuously improve oneself through online courses, e-books, knowledge sharing platforms."},
        {"Creative Expression": "Create text, video, music, art and other works on platforms."},
        {"Exploration and Innovation": "Discover new interests, try new technologies (such as AR, VR)."}
    ],
    "Entertainment and Relaxation Needs": [
        {"Content Consumption": "Watch videos, listen to music, read novels, etc."},
        {"Interactive Entertainment": "Play games, participate in live broadcast interactions."},
        {"Relaxation Experience": "Use meditation, sleep, mental health applications."}
    ],
    "Transaction and Business Needs": [
        {"Online Transactions": "E-commerce shopping, payment transfers."},
        {"Promotion and Marketing": "Promote products or services through advertising, SEO optimization, social media."},
        {"Career and Collaboration": "Find job opportunities, online interviews, cooperation between enterprises."}
    ],
    "Exploration and Curiosity Needs": [
        {"Knowledge Acquisition": "Search encyclopedias, watch popular science videos, participate in Q&A communities."},
        {"Discovering New Things": "Learn about the latest developments through algorithm recommendations and hot lists."},
    ],
    "Belonging and Ritual Needs": [
        {"Digital Identity": "Create and maintain online identities, such as avatars, nicknames, signatures, etc."},
        {"Cultural Participation": "Participate in trending discussions, festival celebrations, major event commemorative activities."},
        {"Virtual Space Belonging": "Have personal 'space' on metaverse or virtual social platforms."}
    ]
}