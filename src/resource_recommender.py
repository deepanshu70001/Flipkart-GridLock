"""
Resource Recommendation Engine
==============================
Converts ML predictions into actionable deployment plans:
- Manpower allocation
- Barricading requirements
- Diversion route suggestions
- Alert level determination
- Cost estimation

This module bridges the gap between ML predictions and
operational traffic management decisions.
"""

import yaml
import numpy as np


class ResourceRecommender:
    """
    Generates actionable resource deployment plans based on ML predictions
    and event characteristics.
    
    Uses a hybrid approach:
    - Rule-based baselines from domain knowledge (config.yaml)
    - ML-informed multipliers from predicted severity/duration/closure
    - Historical correction factors from post-event learning
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.resource_config = self.config["resource_recommender"]
        self.correction_factors = {}  # Updated by post-event learning

    def recommend(self, event_profile, predictions):
        """
        Generate comprehensive resource deployment plan.
        
        Args:
            event_profile: dict with event details
                - event_cause: str
                - corridor: str
                - is_rush_hour: bool
                - is_weekend: bool
                - is_planned: bool
                - latitude, longitude: float
                - junction: str (optional)
            predictions: dict with ML model outputs
                - severity: 'High' or 'Low'
                - severity_proba: float (0-1)
                - duration_hours: dict with 'mean', 'p10', 'p50', 'p90'
                - closure_needed: bool
                - closure_proba: float (0-1)
        
        Returns:
            dict: Complete resource deployment plan
        """
        plan = {
            "alert_level": self._determine_alert_level(predictions),
            "manpower": self._calculate_manpower(event_profile, predictions),
            "barricading": self._calculate_barricading(event_profile, predictions),
            "diversion": self._suggest_diversions(event_profile, predictions),
            "duration_estimate": self._format_duration(predictions),
            "equipment": self._recommend_equipment(event_profile, predictions),
            "communication": self._recommend_communication(event_profile, predictions),
            "estimated_cost": self._estimate_cost(event_profile, predictions),
        }
        return plan

    def _determine_alert_level(self, predictions):
        """
        Determine alert level (1-5) based on combined predictions.
        
        Level 1 (Green):  Low severity, short duration, no closure
        Level 2 (Blue):   Moderate impact
        Level 3 (Yellow): High severity OR long duration OR closure
        Level 4 (Orange): High severity AND (long duration OR closure)
        Level 5 (Red):    High severity AND long duration AND closure
        """
        severity_high = predictions.get("severity_proba", 0.5) > 0.5
        closure = predictions.get("closure_needed", False)
        duration = predictions.get("duration_hours", {}).get("p50", 2)
        long_duration = duration > 6

        score = sum([severity_high, closure, long_duration])

        if score == 0:
            return {"level": 1, "color": "GREEN", "description": "Low impact — standard monitoring"}
        elif score == 1:
            return {"level": 2, "color": "BLUE", "description": "Moderate impact — increased monitoring"}
        elif score == 2:
            return {"level": 3, "color": "YELLOW", "description": "Significant impact — active management required"}
        elif severity_high and closure and not long_duration:
            return {"level": 4, "color": "ORANGE", "description": "High impact — urgent resource deployment"}
        else:
            return {"level": 5, "color": "RED", "description": "Critical — maximum resource deployment"}

    def _calculate_manpower(self, event_profile, predictions):
        """
        Calculate optimal manpower allocation.
        
        Formula:
            Personnel = Base × Severity_Mult × Rush_Mult × Closure_Mult × Correction
        """
        cause = event_profile.get("event_cause", "others")
        base = self.resource_config["manpower_base"].get(cause, 2)

        # Severity multiplier
        severity = "High" if predictions.get("severity_proba", 0.5) > 0.5 else "Low"
        sev_mult = self.resource_config["severity_multiplier"][severity]

        # Rush hour multiplier
        rush_mult = self.resource_config["rush_hour_multiplier"] if event_profile.get("is_rush_hour") else 1.0

        # Road closure multiplier
        closure_mult = self.resource_config["road_closure_multiplier"] if predictions.get("closure_needed") else 1.0

        # Duration-based adjustment
        duration = predictions.get("duration_hours", {}).get("p50", 2)
        if duration > 12:
            duration_mult = 1.5  # Need shift changes
        elif duration > 6:
            duration_mult = 1.2
        else:
            duration_mult = 1.0

        # Correction factor from post-event learning
        correction = self.correction_factors.get(cause, 1.0)

        raw = base * sev_mult * rush_mult * closure_mult * duration_mult * correction
        personnel = int(np.ceil(raw))

        return {
            "total_personnel": personnel,
            "traffic_officers": max(2, int(personnel * 0.4)),
            "constables": int(personnel * 0.4),
            "support_staff": max(1, int(personnel * 0.2)),
            "shifts_needed": max(1, int(np.ceil(duration / 8))),
            "formula": f"Base({base}) × Severity({sev_mult}) × Rush({rush_mult:.1f}) × Closure({closure_mult:.1f}) × Duration({duration_mult:.1f})",
        }

    def _calculate_barricading(self, event_profile, predictions):
        """Calculate barricading plan."""
        closure_needed = predictions.get("closure_needed", False)
        closure_proba = predictions.get("closure_proba", 0)
        cause = event_profile.get("event_cause", "others")

        if closure_needed:
            barricade_type = "full_closure"
            barricade_count = self.resource_config["barricade_base"]["full_closure"]
        elif closure_proba > 0.3:
            barricade_type = "partial_closure"
            barricade_count = self.resource_config["barricade_base"]["partial_closure"]
        else:
            barricade_type = "no_closure"
            barricade_count = self.resource_config["barricade_base"]["no_closure"]

        plan = {
            "type": barricade_type,
            "barricade_count": barricade_count,
            "cones_required": barricade_count * 4,
            "signage_boards": max(2, barricade_count // 2),
        }

        # Event-specific additions
        if cause in ("construction", "public_event", "procession"):
            plan["temporary_fencing_meters"] = 100 if closure_needed else 50
        if cause == "accident":
            plan["reflective_tape_meters"] = 30
            plan["emergency_lighting"] = True

        return plan

    def _suggest_diversions(self, event_profile, predictions):
        """Suggest diversion routes based on corridor."""
        corridor = event_profile.get("corridor", "unknown")
        closure_needed = predictions.get("closure_needed", False)

        # Pre-defined diversion mapping for major Bengaluru corridors
        diversion_map = {
            "Bellary Road 1": {
                "primary": "Palace Road → Sankey Road → Sadashivanagar",
                "secondary": "Cunningham Road → Vasanth Nagar",
                "avoid": "Mekhri Circle during rush hours",
            },
            "Bellary Road 2": {
                "primary": "Hebbal Flyover → ORR North",
                "secondary": "BEL Road → Jalahalli",
                "avoid": "Esteem Mall Junction",
            },
            "Mysore Road": {
                "primary": "Chord Road → Vijayanagar",
                "secondary": "Kanakapura Road via RR Nagar",
                "avoid": "Sirsi Circle during events",
            },
            "Tumkur Road": {
                "primary": "ORR → Yeshwanthpur via Rajajinagar",
                "secondary": "Magadi Road → Chord Road",
                "avoid": "Goraguntepalya Junction",
            },
            "Hosur Road": {
                "primary": "Bannerghatta Road → JP Nagar",
                "secondary": "ORR East via Silk Board",
                "avoid": "Silk Board Junction always",
            },
            "Bannerghata Road": {
                "primary": "Hosur Road → BTM Layout",
                "secondary": "Kanakapura Road → JP Nagar",
                "avoid": "Jayadeva Junction during peak",
            },
            "ORR East 1": {
                "primary": "Old Airport Road → HAL",
                "secondary": "Sarjapur Road → Outer Ring Road",
                "avoid": "Marathahalli Junction",
            },
            "ORR East 2": {
                "primary": "Whitefield Road → ITPL Road",
                "secondary": "Old Madras Road via KR Puram",
                "avoid": "KR Puram Junction during peak",
            },
            "ORR North 1": {
                "primary": "Hennur Road → Banaswadi",
                "secondary": "Thanisandra Road → Nagavara",
                "avoid": "Hebbal Flyover during rush",
            },
            "ORR North 2": {
                "primary": "Hennur Road → Kalyan Nagar → Ramamurthy Nagar",
                "secondary": "Thanisandra Road → Bagalur Road",
                "avoid": "Nagavara Junction during peak",
            },
            "Old Madras Road": {
                "primary": "CMH Road → Indiranagar",
                "secondary": "ORR via KR Puram",
                "avoid": "Trinity Circle",
            },
            "Magadi Road": {
                "primary": "Chord Road → Vijayanagar",
                "secondary": "Tumkur Road via Yeshwanthpur",
                "avoid": "Magadi Road Flyover during construction",
            },
            "West of Chord Road": {
                "primary": "Magadi Road → Rajajinagar Industrial Area",
                "secondary": "Tumkur Road → Yeshwanthpur Circle",
                "avoid": "Mahalakshmi Layout during peak hours",
            },
            "ORR West 1": {
                "primary": "Mysore Road → RR Nagar → Kengeri",
                "secondary": "Chord Road → Vijayanagar → Attiguppe",
                "avoid": "Nayandahalli Junction during peak",
            },
            "CBD 1": {
                "primary": "Residency Road → Richmond Road → Hosur Road",
                "secondary": "MG Road → Trinity Circle → Old Airport Road",
                "avoid": "Corporation Circle during office hours",
            },
            "CBD 2": {
                "primary": "MG Road → Brigade Road → Residency Road",
                "secondary": "JC Road → KR Road → Mysore Road",
                "avoid": "Town Hall area during rallies",
            },
            "Hennur Main Road": {
                "primary": "ORR North → Kalyan Nagar → Banaswadi",
                "secondary": "Thanisandra Road → Jakkur → Yelahanka",
                "avoid": "Hennur Bande Junction during peak",
            },
            "IRR(Thanisandra road)": {
                "primary": "Hennur Road → Hebbal via ORR North",
                "secondary": "Bagalur Road → Yelahanka via NH44",
                "avoid": "Thanisandra Junction during rush hours",
            },
            "Varthur Road": {
                "primary": "Whitefield Main Road → ITPL Road → ORR East",
                "secondary": "Sarjapur Road → Marathahalli via ORR",
                "avoid": "Varthur Kodi Junction during peak",
            },
            "Old Airport Road": {
                "primary": "ORR East → Indiranagar via CMH Road",
                "secondary": "HAL Road → Domlur → Koramangala",
                "avoid": "Manipal Hospital Junction during peak",
            },
            "Airport New South Road": {
                "primary": "Bellary Road → Hebbal Flyover → NH44",
                "secondary": "Hennur Road → Kalyan Nagar → ORR",
                "avoid": "Mekhri Circle during VIP movement",
            },
            "Non-corridor": {
                "primary": "Use nearest parallel arterial road",
                "secondary": "Navigate via adjacent major corridor",
                "avoid": "Event location and 500m radius",
            },
        }

        if corridor in diversion_map:
            diversion = diversion_map[corridor]
        else:
            diversion = {
                "primary": "Use adjacent parallel roads via nearest corridor",
                "secondary": "Check Google Maps for real-time alternatives",
                "avoid": "Event location and 500m radius",
            }

        diversion["closure_needed"] = closure_needed
        diversion["advisory"] = (
            "FULL ROAD CLOSURE — All traffic must divert" if closure_needed
            else "PARTIAL DISRUPTION — Expect delays, consider alternate routes"
        )

        return diversion

    def _format_duration(self, predictions):
        """Format duration predictions for display."""
        dur = predictions.get("duration_hours", {})
        return {
            "expected_hours": round(dur.get("mean", 2), 1),
            "best_case_hours": round(dur.get("p10", 0.5), 1),
            "median_hours": round(dur.get("p50", 2), 1),
            "worst_case_hours": round(dur.get("p90", 12), 1),
            "confidence_interval": f"{dur.get('p25', 1):.1f} – {dur.get('p75', 6):.1f} hours (50% CI)",
        }

    def _recommend_equipment(self, event_profile, predictions):
        """Recommend equipment based on event type."""
        cause = event_profile.get("event_cause", "others")

        base_equipment = ["Walkie-talkies", "Reflective jackets", "Whistles"]

        cause_equipment = {
            "accident": ["First aid kit", "Fire extinguisher", "Tow truck", "Ambulance on standby"],
            "tree_fall": ["Chainsaw", "JCB/crane", "Warning lights"],
            "water_logging": ["Water pump", "Sandbags", "High-visibility markers"],
            "construction": ["Traffic cones", "Diversion signage", "Speed limit boards"],
            "public_event": ["PA system", "Crowd control barriers", "CCTV van"],
            "procession": ["PA system", "Crowd control barriers", "Route markers"],
            "vip_movement": ["Escort vehicles", "Radio communication", "Route clearance team"],
            "vehicle_breakdown": ["Tow truck", "Warning triangle", "Push crew"],
            "protest": ["Crowd control barriers", "Water cannon on standby", "Riot gear"],
        }

        equipment = base_equipment + cause_equipment.get(cause, [])
        return equipment

    def _recommend_communication(self, event_profile, predictions):
        """Recommend communication actions."""
        alert = self._determine_alert_level(predictions)
        level = alert["level"]

        actions = []
        if level >= 1:
            actions.append("Update control room dashboard")
        if level >= 2:
            actions.append("Notify nearby patrol units")
            actions.append("Update Google Maps / Waze via API")
        if level >= 3:
            actions.append("Send SMS alerts to registered commuters")
            actions.append("Post on social media (Twitter/X)")
            actions.append("Notify local media")
        if level >= 4:
            actions.append("Activate emergency response protocol")
            actions.append("Notify senior officers / ACP")
        if level >= 5:
            actions.append("Notify Commissioner's office")
            actions.append("Request military / NDRF assistance if needed")

        return {"actions": actions, "priority": alert["color"]}

    def _estimate_cost(self, event_profile, predictions):
        """Estimate deployment cost."""
        manpower = self._calculate_manpower(event_profile, predictions)
        barricading = self._calculate_barricading(event_profile, predictions)
        duration = predictions.get("duration_hours", {}).get("p50", 2)

        # Cost estimates (INR per unit per hour)
        officer_rate = 500
        constable_rate = 300
        support_rate = 200
        barricade_rate = 50  # per barricade per deployment

        manpower_cost = (
            manpower["traffic_officers"] * officer_rate * duration
            + manpower["constables"] * constable_rate * duration
            + manpower["support_staff"] * support_rate * duration
        )
        barricade_cost = barricading["barricade_count"] * barricade_rate
        equipment_cost = 2000 if predictions.get("closure_needed") else 500

        total = manpower_cost + barricade_cost + equipment_cost

        return {
            "manpower_cost_inr": round(manpower_cost),
            "barricade_cost_inr": round(barricade_cost),
            "equipment_cost_inr": round(equipment_cost),
            "total_estimated_cost_inr": round(total),
            "total_formatted": f"₹{total:,.0f}",
        }

    def update_correction_factors(self, factors):
        """Update correction factors from post-event learning."""
        self.correction_factors.update(factors)
        print(f"[ResourceRecommender] Updated correction factors for {len(factors)} event causes")
