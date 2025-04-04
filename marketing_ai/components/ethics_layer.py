from transformers import pipeline
from typing import Dict, Any, Optional, List
import json

class MarketingEthicsLayer:
    def __init__(self):
        self.ethics_nlp = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.toxicity_detector = pipeline("text-classification", model="unitary/multilingual-toxic-xlm-roberta")
        self.bias_detector = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def check_marketing_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check marketing content for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": [],
            "detailed_analysis": {}
        }

        # Check text content
        if "text" in content:
            text_analysis = self._analyze_text(content["text"])
            results["detailed_analysis"]["text"] = text_analysis
            if not text_analysis["is_ethical"]:
                results["is_ethical"] = False
                results["warnings"].extend(text_analysis["warnings"])
                results["suggestions"].extend(text_analysis["suggestions"])

        # Check targeting
        if "target_audience" in content:
            targeting_analysis = self._analyze_targeting(content["target_audience"])
            results["detailed_analysis"]["targeting"] = targeting_analysis
            if not targeting_analysis["is_ethical"]:
                results["is_ethical"] = False
                results["warnings"].extend(targeting_analysis["warnings"])
                results["suggestions"].extend(targeting_analysis["suggestions"])

        # Check claims
        if "claims" in content:
            claims_analysis = self._analyze_claims(content["claims"])
            results["detailed_analysis"]["claims"] = claims_analysis
            if not claims_analysis["is_ethical"]:
                results["is_ethical"] = False
                results["warnings"].extend(claims_analysis["warnings"])
                results["suggestions"].extend(claims_analysis["suggestions"])

        return results

    def check_campaign_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Check marketing campaign strategy for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": [],
            "detailed_analysis": {}
        }

        # Check targeting strategy
        if "targeting" in strategy:
            targeting_analysis = self._analyze_targeting_strategy(strategy["targeting"])
            results["detailed_analysis"]["targeting"] = targeting_analysis
            if not targeting_analysis["is_ethical"]:
                results["is_ethical"] = False
                results["warnings"].extend(targeting_analysis["warnings"])
                results["suggestions"].extend(targeting_analysis["suggestions"])

        # Check messaging strategy
        if "messaging" in strategy:
            messaging_analysis = self._analyze_messaging_strategy(strategy["messaging"])
            results["detailed_analysis"]["messaging"] = messaging_analysis
            if not messaging_analysis["is_ethical"]:
                results["is_ethical"] = False
                results["warnings"].extend(messaging_analysis["warnings"])
                results["suggestions"].extend(messaging_analysis["suggestions"])

        # Check channel strategy
        if "channels" in strategy:
            channel_analysis = self._analyze_channel_strategy(strategy["channels"])
            results["detailed_analysis"]["channels"] = channel_analysis
            if not channel_analysis["is_ethical"]:
                results["is_ethical"] = False
                results["warnings"].extend(channel_analysis["warnings"])
                results["suggestions"].extend(channel_analysis["suggestions"])

        return results

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text content for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": []
        }

        # Check for toxicity
        toxicity = self.toxicity_detector(text)[0]
        if toxicity["label"] == "toxic" and toxicity["score"] > 0.5:
            results["is_ethical"] = False
            results["warnings"].append("Content contains potentially toxic language")
            results["suggestions"].append("Review and revise language to be more inclusive and respectful")

        # Check for bias
        bias_categories = ["gender", "race", "age", "socioeconomic"]
        bias_results = self.bias_detector(text, candidate_labels=bias_categories)
        for label, score in zip(bias_results["labels"], bias_results["scores"]):
            if score > 0.7:
                results["is_ethical"] = False
                results["warnings"].append(f"Potential {label} bias detected")
                results["suggestions"].append(f"Review content for {label} inclusivity")

        return results

    def _analyze_targeting(self, targeting: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze targeting criteria for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": []
        }

        # Check for discriminatory targeting
        if "demographics" in targeting:
            demographics = targeting["demographics"]
            if "age" in demographics:
                age_ranges = demographics["age"]
                if any(age < 13 for age in age_ranges):
                    results["is_ethical"] = False
                    results["warnings"].append("Targeting includes underage audience")
                    results["suggestions"].append("Remove underage targeting")

            if "income" in demographics:
                income_levels = demographics["income"]
                if "low" in income_levels:
                    results["warnings"].append("Targeting low-income groups may raise ethical concerns")
                    results["suggestions"].append("Consider inclusive pricing strategies")

        # Check for manipulative targeting
        if "behavioral" in targeting:
            behavioral = targeting["behavioral"]
            if "vulnerability" in behavioral:
                results["is_ethical"] = False
                results["warnings"].append("Targeting based on vulnerability is unethical")
                results["suggestions"].append("Remove vulnerability-based targeting")

        return results

    def _analyze_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Analyze marketing claims for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": []
        }

        for claim in claims:
            # Check for exaggerated claims
            if any(word in claim.lower() for word in ["best", "perfect", "guaranteed", "miracle"]):
                results["warnings"].append(f"Potential exaggeration in claim: {claim}")
                results["suggestions"].append("Use more specific, verifiable language")

            # Check for misleading claims
            if any(word in claim.lower() for word in ["free", "limited time", "exclusive"]):
                results["warnings"].append(f"Potential misleading claim: {claim}")
                results["suggestions"].append("Add clear terms and conditions")

        return results

    def _analyze_targeting_strategy(self, targeting: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze targeting strategy for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": []
        }

        # Check for exclusionary targeting
        if "exclude" in targeting:
            excluded_groups = targeting["exclude"]
            if any(group in excluded_groups for group in ["protected_classes"]):
                results["is_ethical"] = False
                results["warnings"].append("Exclusion of protected classes is unethical")
                results["suggestions"].append("Remove exclusionary targeting")

        # Check for data privacy compliance
        if "data_usage" in targeting:
            data_usage = targeting["data_usage"]
            if not data_usage.get("consent_obtained", False):
                results["is_ethical"] = False
                results["warnings"].append("Missing user consent for data usage")
                results["suggestions"].append("Implement proper consent mechanisms")

        return results

    def _analyze_messaging_strategy(self, messaging: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze messaging strategy for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": []
        }

        # Check for manipulative messaging
        if "tone" in messaging:
            tone = messaging["tone"]
            if tone in ["fear", "guilt", "pressure"]:
                results["is_ethical"] = False
                results["warnings"].append("Manipulative messaging tone detected")
                results["suggestions"].append("Use more positive, empowering messaging")

        # Check for transparency
        if "disclosure" not in messaging:
            results["warnings"].append("Missing disclosure information")
            results["suggestions"].append("Add clear disclosure statements")

        return results

    def _analyze_channel_strategy(self, channels: List[str]) -> Dict[str, Any]:
        """Analyze channel strategy for ethical compliance."""
        results = {
            "is_ethical": True,
            "warnings": [],
            "suggestions": []
        }

        # Check for age-appropriate channels
        if "tiktok" in channels:
            results["warnings"].append("TikTok may expose content to underage users")
            results["suggestions"].append("Implement age verification")

        # Check for privacy concerns
        if "whatsapp" in channels:
            results["warnings"].append("WhatsApp requires careful privacy handling")
            results["suggestions"].append("Review WhatsApp privacy policies")

        return results

# Example usage
if __name__ == "__main__":
    ethics = MarketingEthicsLayer()
    
    # Test content analysis
    content = {
        "text": "Our product is the best in the world! Limited time offer!",
        "target_audience": {
            "demographics": {
                "age": [18, 65],
                "income": ["low", "medium"]
            }
        },
        "claims": [
            "Best product ever",
            "Free shipping",
            "Limited time offer"
        ]
    }
    
    content_analysis = ethics.check_marketing_content(content)
    print("Content Analysis:", json.dumps(content_analysis, indent=2))
    
    # Test campaign strategy
    strategy = {
        "targeting": {
            "exclude": ["protected_classes"],
            "data_usage": {
                "consent_obtained": False
            }
        },
        "messaging": {
            "tone": "fear"
        },
        "channels": ["facebook", "tiktok", "whatsapp"]
    }
    
    strategy_analysis = ethics.check_campaign_strategy(strategy)
    print("Strategy Analysis:", json.dumps(strategy_analysis, indent=2)) 