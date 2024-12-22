# Import necessary libraries for FastAPI, file handling, JSON parsing, and NLP
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import re

# Initialize FastAPI application
app = FastAPI()

# Configure logging to display informational messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP models once for efficiency
# Note: Placeholder model, consider using a model more suited for clinical contexts
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                              revision="714eb0f")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


class AssessmentAgent:
    def __init__(self, assessment_name):
        # Initialize with the name of the assessment and an empty scoring map
        self.assessment_name = assessment_name
        self.scoring_map = {}

    def analyze(self, session_data):
        # Calculate score based on symptoms described in session data
        score = 0
        for symptom in session_data['Psychological Factors']['Symptoms'].values():
            for key in self.scoring_map:
                if key.lower() in symptom['Description'].lower():
                    score += self.scoring_map[key]
                    break
        return score


class GAD7Agent(AssessmentAgent):
    def __init__(self):
        # Initialize with GAD-7 criteria
        super().__init__("GAD-7")
        self.scoring_map = {
            "feeling nervous, anxious, or on edge": 1,
            "not being able to stop or control worrying": 1,
            "worrying too much about different things": 1,
            "trouble relaxing": 1,
            "being so restless that it's hard to sit still": 1,
            "becoming easily annoyed or irritable": 1,
            "feeling afraid as if something awful might happen": 1
        }


class PHQ9Agent(AssessmentAgent):
    def __init__(self):
        # Initialize with PHQ-9 criteria
        super().__init__("PHQ-9")
        self.scoring_map = {
            "little interest or pleasure in doing things": 1,
            "feeling down, depressed, or hopeless": 1,
            "trouble falling or staying asleep, or sleeping too much": 1,
            "feeling tired or having little energy": 1,
            "poor appetite or overeating": 1,
            "feeling bad about yourself - or that you are a failure or have let yourself or your family down": 1,
            "trouble concentrating on things, such as reading the newspaper or watching television": 1,
            "moving or speaking so slowly that other people could have noticed": 1,
            "thoughts that you would be better off dead, or of hurting yourself": 1
        }


class AssessmentRouter:
    def select_agent(self, session_data):
        # Choose the appropriate agent based on session content
        summary = session_data['Brief Summary of Session'].lower()
        if "anxiety" in summary:
            return GAD7Agent()
        elif "depression" in summary:
            return PHQ9Agent()
        return None  # or default agent if none match

    def process_session(self, session_data):
        # Process session data with the selected agent
        agent = self.select_agent(session_data)
        if agent:
            return agent.analyze(session_data)
        return "No suitable assessment found."


def semantic_similarity(text1, text2):
    # Compute semantic similarity between two texts using Sentence-BERT
    embeddings1 = sbert_model.encode([text1])
    embeddings2 = sbert_model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return max(0, similarity)  # Ensure non-negative similarity score


def is_narrative_improved(old_value, new_value):
    # Determine if narrative has improved based on keywords and semantics
    if old_value == "NA" and new_value != "NA":
        return True
    if old_value == new_value == "NA":
        return False

    positive_keywords = ["improved", "better", "balanced", "empowering", "valued", "mindful", "consistent",
                         "reconnected", "positive", "less", "fewer", "reduced", "eased"]
    negative_keywords = ["difficult", "disappoint", "undervalued", "resentful", "neglected", "disconnected", "anxious",
                         "stress", "struggling", "overwhelm", "isolated", "worsened", "more", "higher", "increased"]

    new_value_no_numbers = re.sub(r'\d+', '', new_value)

    # Check for positive or negative keywords, considering negations
    positive = any(keyword in new_value_no_numbers.lower() for keyword in positive_keywords if
                   not any(neg in new_value_no_numbers.lower() for neg in ["not", "no", "less"]))
    negative = any(keyword in new_value_no_numbers.lower() for keyword in negative_keywords if
                   not any(neg in new_value_no_numbers.lower() for neg in ["not", "no", "less"]))

    if positive and not negative:
        return True
    if negative and not positive:
        return False

    # Fallback to semantic similarity if no clear indicator from keywords
    return semantic_similarity(old_value, new_value) > 0.5


def analyse_progress(data1, data2):
    # Analyze progress between two sessions
    progress = {}
    router = AssessmentRouter()

    # Compare symptoms
    symptoms1 = data1['Psychological Factors']['Symptoms']
    symptoms2 = data2['Psychological Factors']['Symptoms']
    for symptom, details in symptoms1.items():
        if symptom in symptoms2:
            sentiment1 = sentiment_analyzer(details['Description'])[0]['score']
            sentiment2 = sentiment_analyzer(symptoms2[symptom]['Description'])[0]['score']
            similarity = semantic_similarity(details['Description'], symptoms2[symptom]['Description'])
            narrative_improvement = is_narrative_improved(details['Description'], symptoms2[symptom]['Description'])

            # Score symptoms using agents
            agent_score1 = router.process_session(data1)
            agent_score2 = router.process_session(data2)

            if isinstance(agent_score1, int) and isinstance(agent_score2, int):
                agent_improvement = agent_score2 < agent_score1  # Lower score indicates improvement
            else:
                agent_improvement = None

            # Determine progress status, giving narrative precedence in case of conflict
            progress_status = "Improved" if (
                        narrative_improvement or (agent_improvement is not None and agent_improvement)) else \
                "Declined" if (
                            not narrative_improvement or (agent_improvement is not None and not agent_improvement)) else \
                    "Plateaued"

            if narrative_improvement != agent_improvement and agent_improvement is not None:
                progress_status += " (Note: Agent score suggests different progress)"

            progress[symptom] = {
                'description': details['Description'],
                'old_sentiment': sentiment1,
                'new_sentiment': sentiment2,
                'similarity': similarity,
                'agent_score_old': agent_score1 if isinstance(agent_score1, int) else "No Score",
                'agent_score_new': agent_score2 if isinstance(agent_score2, int) else "No Score",
                'progress': progress_status
            }

    # Compare other fields in categories
    categories = {
        'Biological Factors': ['Sleep', 'Nutrition', 'Physical Activity'],
        'Social Factors': ['Work or School', 'Relationships', 'Recreation'],
        'Mental Status Exam': ['Mood and Affect', 'Speech and Language', 'Thought Process and Content']
    }

    for category, fields in categories.items():
        for field in fields:
            if field in data1[category] and field in data2[category]:
                old_value = data1[category][field]
                new_value = data2[category][field]

                similarity = semantic_similarity(old_value, new_value)

                if old_value == 'NA' and new_value != 'NA':
                    progress_status = "Improved"
                elif new_value == 'NA' and old_value != 'NA':
                    progress_status = "Declined"
                else:
                    narrative_improvement = is_narrative_improved(old_value, new_value)
                    progress_status = "Improved" if similarity > 0.5 and narrative_improvement else \
                        "Declined" if similarity < 0.3 or not narrative_improvement else \
                            "Plateaued"

                progress[field] = {
                    'old_value': old_value,
                    'new_value': new_value,
                    'similarity': similarity,
                    'progress': progress_status
                }

    # Add overall assessment score comparison
    score1 = router.process_session(data1)
    score2 = router.process_session(data2)

    if isinstance(score1, int) and isinstance(score2, int):
        progress['Assessment'] = {
            'old_score': score1,
            'new_score': score2,
            'progress': "Improved" if score2 < score1 else "Declined" if score2 > score1 else "Stable"
        }

    return progress


def convert_numpy_to_float(data):
    # Convert numpy float32 to Python float for JSON serialization
    if isinstance(data, dict):
        return {k: convert_numpy_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_float(elem) for elem in data]
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data


@app.post("/compare_sessions/")
async def compare_sessions(session1: UploadFile = File(...), session2: UploadFile = File(...)):
    try:
        content1 = await session1.read()
        content2 = await session2.read()

        try:
            data1 = json.loads(content1.decode('utf-8'))
            data2 = json.loads(content2.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format in the uploaded file: {str(e)}")

        logger.info("Successfully parsed session data")

        progress = analyse_progress(data1, data2)
        progress = convert_numpy_to_float(progress)

        logger.info(f"Progress data before return: {progress}")
        return {"progress": progress}
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing files: {str(e)}")