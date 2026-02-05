import json
import os
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from google import genai
from google.genai import types
from groq import Groq
from rouge_score import rouge_scorer
import asyncio
from concurrent.futures import ThreadPoolExecutor , as_completed   
import threading

# ============================================================
# THREAD-SAFE OUTPUT HELPERS (NEW)
# ============================================================

write_lock = threading.Lock()


def load_existing_output(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠ Output JSON corrupted. Starting fresh.")
                return {}
    return {}


def append_result(pair_uid: str, result: Dict, output_file: str):
    with write_lock:
        data = load_existing_output(output_file)
        data[pair_uid] = result
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# ============================================================
# INTENSITY EVALUATOR AGENT — SUPPORTS MULTIPLE LLM PROVIDERS
# ============================================================
class IntensityEvaluatorAgent:
    def __init__(self, agent_id: str, provider: str, api_key: str, model: str):
        """
        Initialize agent with specific LLM provider.
        
        Args:
            agent_id: Identifier for this agent
            provider: One of "openai", "gemini", or "groq"
            api_key: API key for the provider
            model: Model name to use
        """
        self.agent_id = agent_id
        self.provider = provider.lower()
        self.model = model
        self.temperature = 0
        
        # Initialize the appropriate client
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=api_key)
        elif self.provider == "groq":
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'gemini', or 'groq'")

        self.intensity_instruction = """
You are an intensity scorer for peer-review POTENTIAL contradictions.

INTENSITY LEVELS:
- Score 0 — No Contradiction (Compatible or Orthogonal Statements)
Statements refer to different aspects, topics, or evaluation criteria, OR
Statements discuss the same aspect but are fully compatible or consistent, OR
Any differences are descriptive, complementary, or additive rather than conflicting.
EXAMPLES:-
A. R1: "The paper is clearly written and easy to understand."
   R2: "The writing is mostly clear, with well-structured sections."
B. R1: "The paper introduces a novel perspective on contradiction detection."
   R2: "The experimental setup is clearly described and easy to follow."

- Score 1 — Low Severity (Implicit Contradiction) One statement is generic, the other is specific, OR The conflict is indirect or interpretative, No strong positive/negative polarity, Weak or implicit disagreement.
EXAMPLES:-
A. "review1": "Section 3, where the authors describe the proposed techniques is somewhat confusing to read, because of a lack of detailed mathematical explanations of the proposed techniques.",
   "review2": "The paper is clearly written and results seem compelling."
B. "review1": "The classification function stays unfortunately trivial (and classical in graph-based problem)...",
    "review2": "The explanation of the method gives a clear intuition on why the proposed method makes sense to find adversarially robust features."

- Score 2 — Moderate Severity (Explicit but Mild Conflict) Both statements explicitly refer to the same aspect, One gives light criticism, the other is mildly or significantly positive, Explicit but not extreme polarity.
Examples:-
A. "review1": "I had trouble following the details of the DHS Softmax, however.",
    "review2": "The resulting algorithm is simple and doesn't seem to entail a significant computational burden."
B. "review1": "However, the novelty is limited in the sense it is application of coordinate descent on power iterations.",
    "review2": "This paper appears to be the first to solve this problem, and make a connection to coordinate decent."

- Score 3 — High Severity (Direct Strong Contradiction) Strongly worded positive vs. negative evaluation of the same aspect, Extremely polarized opposite judgments, Clear and fundamental extreme disagreement.
EXAMPLES:-
A. "The paper is clearly written.",
    "I found the presentation of the proposed measure overly confusing."
B. "Review 1": "This decreases the credibility of the simulation that shows it outperforms the baselines such as TCAV, as it sounds like you set the parameters to get good metrics on this simulation.",
    "Review 2": "The method is finally tested on a variety of datasets, including a synthetic one, which shows that optimizing 'completeness' helps in discovering a richer variety of important concepts than prior work.
"""

    def initial_score(self, evidence_pair: List[str], review_1: str, review_2: str) -> Tuple[int, str]:
        """Provide initial intensity score."""
        if not isinstance(evidence_pair, list) or len(evidence_pair) != 2:
            return 0, "Invalid evidence format"

        s1, s2 = evidence_pair
        
        prompt = f"""
{self.intensity_instruction}

REVIEW 1 CONTEXT:
\"\"\"{review_1}\"\"\"

REVIEW 2 CONTEXT:
\"\"\"{review_2}\"\"\"

EVIDENCE FROM REVIEW 1:
\"\"\"{s1}\"\"\"

EVIDENCE FROM REVIEW 2:
\"\"\"{s2}\"\"\"

Analyze these two pieces of evidence and determine the intensity of contradiction.

Output your response in this JSON format:
{{
  "intensity": <0, 1, 2, or 3>,
  "reasoning": "detailed explanation of why you assigned this intensity score"
}}

Output ONLY the JSON object, nothing else.
"""
        return self._make_api_call(prompt)

    def debate_response(self, evidence_pair: List[str], review_1: str, review_2: str, 
                       my_score: int, my_reasoning: str, 
                       opponent_score: int, opponent_reasoning: str,
                       conversation_history: List[Dict]) -> Tuple[int, str]:
        """Generate a debate response defending your score. Score CANNOT be changed."""
        s1, s2 = evidence_pair
        
        # Build conversation context
        debate_context = "\n\n".join([
            f"[Round {msg['round']}] {msg['agent']}: Score {msg['score']}\n{msg['reasoning']}"
            for msg in conversation_history
        ])
        
        prompt = f"""
{self.intensity_instruction}

REVIEW 1 CONTEXT:
\"\"\"{review_1}\"\"\"

REVIEW 2 CONTEXT:
\"\"\"{review_2}\"\"\"

EVIDENCE FROM REVIEW 1:
\"\"\"{s1}\"\"\"

EVIDENCE FROM REVIEW 2:
\"\"\"{s2}\"\"\"

DEBATE HISTORY:
{debate_context}

YOUR ASSIGNED SCORE: {my_score}
YOUR PREVIOUS REASONING: {my_reasoning}

OPPONENT'S SCORE: {opponent_score}
OPPONENT'S REASONING: {opponent_reasoning}

CRITICAL INSTRUCTION: You MUST defend your score of {my_score}. You CANNOT change your score during the debate.

RULES FOR EVIDENCE-BASED DEBATE:
1. ONLY cite text that actually appears in the evidence or reviews above
2. Use direct quotes (in "quotation marks") when referencing the reviews
3. Point to specific words, phrases, or sentences that support your position
4. Counter your opponent by showing what they missed or misinterpreted in the actual text
5. Use simple language - avoid vague terms like "nuanced", "multifaceted", "complex interplay".
6. Don't use bold word or italics formatting in your response.
6. If opponent makes claims not supported by the text, point this out specifically

Your task:
1. Quote specific phrases from the evidence that support your score of {my_score}
2. Identify flaws in opponent's reasoning by showing what the text actually says
4. Explain why your intensity level {my_score} fits the criteria better than {opponent_score}

Be assertive but fair. Focus on why YOUR intensity assessment is correct.

Output your response in this JSON format:
{{
  "intensity": {my_score},
  "reasoning": "your defense of score {my_score} and counterarguments against opponent's score {opponent_score}"
}}

IMPORTANT: The intensity field MUST be exactly {my_score}. Do not change it
Output ONLY the JSON object, no other characters beyond the JSON.
"""
        result_score, result_reasoning = self._make_api_call(prompt)
        
        # Enforce that score cannot change during debate
        if result_score != my_score:
            print(f"        Warning: {self.agent_id} attempted to change score from {my_score} to {result_score}. Reverting.")
            result_score = my_score
        
        return result_score, result_reasoning

    def _make_api_call(self, prompt: str) -> Tuple[int, str]:
        """Make API call to the appropriate provider and parse response."""
        max_retries = 1  # Number of retries for API calls
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.choices[0].message.content.strip()
        
                elif self.provider == "gemini":
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=2048,
                        )
                    )
                    content = response.text.strip()
        
                elif self.provider == "groq":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.choices[0].message.content.strip()
            
                # Remove triple backticks if present
                if content.startswith("```") and content.endswith("```"):
                    content = content[3:-3].strip()
                content = re.sub(r'^json', '', content, flags=re.IGNORECASE).strip()
            
                # Parse JSON response
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"        {self.agent_id} ({self.provider}) JSON parsing error: {str(e)}")
                    # Attempt to sanitize the response
                    sanitized_content = re.sub(r'[\x00-\x1F\x7F]', '', content)  # Remove control characters
                    sanitized_content = re.sub(r'```(?:json)?', '', sanitized_content).strip()  # Remove backticks
                    try:
                        result = json.loads(sanitized_content)
                    except json.JSONDecodeError as e:
                        print(f"        {self.agent_id} ({self.provider}) Failed to parse sanitized JSON: {str(e)}")
                        raise ValueError("Malformed response after sanitization")
            
            # Extract intensity and reasoning
                intensity = result.get("intensity", 0)
                reasoning = result.get("reasoning", "No reasoning provided")
                if intensity == 0 and reasoning == "No reasoning provided":
                    print(f"        {self.agent_id} ({self.provider}) Warning: Response missing expected fields.")
                if not isinstance(intensity, int) or intensity not in [0, 1, 2, 3]:
                    raise ValueError("Invalid intensity value in response")
            
                return intensity, reasoning  # Return valid response if no exception occurs

            except Exception as e:
                print(f"        {self.agent_id} ({self.provider}) API error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"        {self.agent_id} ({self.provider}) Max retries reached. Failing.")
                    return 0, "Failed to get a valid response"
    
        return 0, "Failed to get a valid response"


# ============================================================
# JUDGE AGENT — MAKES FINAL DECISION BASED ON DEBATE
# ============================================================
class JudgeAgent:
    def __init__(self, provider: str, api_key: str, model: str):
        """
        Initialize judge with specific LLM provider.
        
        Args:
            provider: One of "openai", "gemini", or "groq"
            api_key: API key for the provider
            model: Model name to use
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = 0
        
        # Initialize the appropriate client
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=api_key)  # Force AI Studio API
        elif self.provider == "groq":
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.judge_instruction = """
You are a judge evaluating a debate between two intensity scorers for peer-review contradictions.

INTENSITY LEVELS:
- Score 0 — No Contradiction (Compatible or Orthogonal Statements)
Statements refer to different aspects, topics, or evaluation criteria, OR
Statements discuss the same aspect but are fully compatible or consistent, OR
Any differences are descriptive, complementary, or additive rather than conflicting.

- Score 1 — Low Severity (Implicit Contradiction) One statement is generic, the other is specific, OR The conflict is indirect or interpretative, No strong positive/negative polarity, Weak or implicit disagreement.

- Score 2 — Moderate Severity (Explicit but Mild Conflict) Both statements explicitly refer to the same aspect, One gives light criticism, the other is mildly or significantly positive, Explicit but not extreme polarity.

- Score 3 — High Severity (Direct Strong Contradiction) Strongly worded positive vs. negative evaluation of the same aspect, Extremely polarized opposite judgments, Clear and fundamental extreme disagreement.

Your task is to:
1. Review the entire debate conversation
2. Examine the evidence from both reviews
3. Make an final judgment on which of the two intensity scores is correct based on the reviews and debate between the agents.
4. Provide clear reasoning for your decision.
5. You MUST chose one of the two intensity scores presented by the agents.

Consider:
- Which arguments were most convincing?
- What does the evidence actually show?
- Does the contradiction meet the criteria for the claimed intensity level?
- Your decision must be based solely on the evidence and debate provided.
"""

    def make_final_judgment(self, evidence_pair: List[str], review_1: str, review_2: str,
                           conversation_history: List[Dict]) -> Tuple[int, str]:
        """Make final judgment after reviewing the debate."""
        s1, s2 = evidence_pair
        
        # Build debate summary
        debate_summary = "\n\n".join([
            f"[Round {msg['round']}] {msg['agent']}: Score {msg['score']}\n{msg['reasoning']}"
            for msg in conversation_history
        ])
        
        prompt = f"""
{self.judge_instruction}

REVIEW 1 CONTEXT:
\"\"\"{review_1}\"\"\"

REVIEW 2 CONTEXT:
\"\"\"{review_2}\"\"\"

EVIDENCE FROM REVIEW 1:
\"\"\"{s1}\"\"\"

EVIDENCE FROM REVIEW 2:
\"\"\"{s2}\"\"\"

COMPLETE DEBATE TRANSCRIPT:
{debate_summary}

Based on the evidence and the debate, make your final judgment on the intensity score.

Output your response in this JSON format:
{{
  "intensity": <0, 1, 2, or 3>,
  "reasoning": "your final judgment explaining why this is the correct intensity score"
}}

Output ONLY the JSON object, nothing else.
"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                
            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=2048,
                    )
                )
                content = response.text.strip()
                
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    return 0, "Failed to parse response"
            
            intensity = result.get("intensity", 0)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            if not isinstance(intensity, int) or intensity not in [0, 1, 2, 3]:
                return 0, "Invalid intensity value"
            
            # Determine which agent the judge agreed with
            agent1_score = conversation_history[0]['score']  # Agent_1's initial score
            agent2_score = conversation_history[1]['score']  # Agent_2's initial score

            judge_agreed_with = None
            if intensity == agent1_score and intensity != agent2_score:
                judge_agreed_with = "Agent_1"
            elif intensity == agent2_score and intensity != agent1_score:
                judge_agreed_with = "Agent_2"
            elif intensity == agent1_score and intensity == agent2_score:
                judge_agreed_with = "Both"
            else:
                judge_agreed_with = "Neither" 
            return intensity, reasoning, judge_agreed_with
            
        except Exception as e:
            print(f"        Judge ({self.provider}) API error: {str(e)}")
            return 0, f"Error: {str(e)}"


# ============================================================
# MULTI-AGENT DEBATE SCORER WITH MULTIPLE PROVIDERS
# ============================================================
class MultiAgentDebateScorer:
    def __init__(self, 
                 agent1_config: Dict,
                 agent2_config: Dict,
                 judge_config: Dict):
        """
        Initialize multi-agent debate scorer with different providers.
        
        Args:
            agent1_config: {"provider": "gemini", "api_key": "...", "model": "gemini-1.5-flash"}
            agent2_config: {"provider": "groq", "api_key": "...", "model": "llama-3.1-70b-versatile"}
            judge_config: {"provider": "openai", "api_key": "...", "model": "gpt-4o"}
        """
        # Initialize two evaluator agents
        self.agent1 = IntensityEvaluatorAgent(
            "Agent_1", 
            agent1_config["provider"],
            agent1_config["api_key"],
            agent1_config["model"]
        )
        self.agent2 = IntensityEvaluatorAgent(
            "Agent_2",
            agent2_config["provider"],
            agent2_config["api_key"],
            agent2_config["model"]
        )
        
        # Initialize judge
        self.judge = JudgeAgent(
            judge_config["provider"],
            judge_config["api_key"],
            judge_config["model"]
        )
        
        self.max_debate_rounds = 4

    def score_with_debate(self, evidence_pair: List[str], review_1: str, review_2: str) -> Dict:
        """
        Score intensity with multi-agent debate system.
        Returns final score, reasoning, and debate history.
        """
        if not isinstance(evidence_pair, list) or len(evidence_pair) != 2:
            return {
                "intensity": 0,
                "reasoning": "Invalid evidence format",
                "debate_history": [],
                "agreement": True
            }

        # Phase 1: Initial scoring by both agents
        score1, reasoning1 = self.agent1.initial_score(evidence_pair, review_1, review_2)
        score2, reasoning2 = self.agent2.initial_score(evidence_pair, review_1, review_2)
        
        conversation_history = [
            {"round": 1, "agent": "Agent_1", "score": score1, "reasoning": reasoning1},
            {"round": 1, "agent": "Agent_2", "score": score2, "reasoning": reasoning2}
        ]
        
        print(f"          Initial scores: Agent_1={score1}, Agent_2={score2}")
        
        # If agents agree, return immediately
        if score1 == score2:
            print(f"          ✓ Agreement reached immediately")
            return {
                "intensity": score1,
                "reasoning": reasoning1,
                "debate_history": conversation_history,
                "agreement": True,
                "rounds": 1
            }
        
        # Phase 2: Debate for up to max_debate_rounds
        print(f"          ✗ Disagreement - Starting debate (scores locked)...")
        current_score1, current_reasoning1 = score1, reasoning1
        current_score2, current_reasoning2 = score2, reasoning2
        
        for round_num in range(2, self.max_debate_rounds + 2):
            print(f"          Debate Round {round_num}...")
            
            # Agent 1 defends their score (score cannot change)
            returned_score1, current_reasoning1 = self.agent1.debate_response(
                evidence_pair, review_1, review_2,
                current_score1, current_reasoning1,
                current_score2, current_reasoning2,
                conversation_history
            )
            # Verify score didn't change
            assert returned_score1 == current_score1, f"Agent 1 score changed illegally: {current_score1} -> {returned_score1}"
            
            conversation_history.append({
                "round": round_num,
                "agent": "Agent_1",
                "score": current_score1,
                "reasoning": current_reasoning1
            })
            
            # Agent 2 defends their score (score cannot change)
            returned_score2, current_reasoning2 = self.agent2.debate_response(
                evidence_pair, review_1, review_2,
                current_score2, current_reasoning2,
                current_score1, current_reasoning1,
                conversation_history
            )
            # Verify score didn't change
            assert returned_score2 == current_score2, f"Agent 2 score changed illegally: {current_score2} -> {returned_score2}"
            
            conversation_history.append({
                "round": round_num,
                "agent": "Agent_2",
                "score": current_score2,
                "reasoning": current_reasoning2
            })
            
            print(f"          Round {round_num} scores: Agent_1={current_score1}, Agent_2={current_score2} (locked)")
        
        # After max rounds, scores are still different - judge must decide
        print(f"          Scores remain: Agent_1={current_score1}, Agent_2={current_score2} - Judge deciding...")
        final_score, final_reasoning, judge_agreed_with = self.judge.make_final_judgment(
            evidence_pair, review_1, review_2, conversation_history
        )
        
        print(f"          ⚖ Judge's final decision: {final_score}")
        
        conversation_history.append({
            "round": "FINAL",
            "agent": "Judge",
            "score": final_score,
            "reasoning": final_reasoning,
            "judge_agreed_with": judge_agreed_with
        })
        
        return {
            "intensity": final_score,
            "reasoning": final_reasoning,
            "debate_history": conversation_history,
            "agreement": False,
            "rounds": self.max_debate_rounds + 1,
            "judge_agreed_with": judge_agreed_with 
        }


# ============================================================
# EXTERNAL ITERATIVE DETECTOR WITH MULTI-AGENT DEBATE
# ============================================================
class ExternalIterativeDetector:
    def __init__(self, 
                 generator_config: Dict,
                 agent1_config: Dict,
                 agent2_config: Dict,
                 judge_config: Dict):
        """
        Initialize the detector with multiple LLM providers.
        
        Args:
            generator_config: {"provider": "openai", "api_key": "...", "model": "gpt-4o-mini"}
            agent1_config: Config for first evaluator agent
            agent2_config: Config for second evaluator agent
            judge_config: Config for judge agent
        """
        self.generator_provider = generator_config["provider"].lower()
        self.generator_model = generator_config["model"]
        self.temperature = 0
        
        # Initialize generator client
        if self.generator_provider == "openai":
            self.client = OpenAI(api_key=generator_config["api_key"])
        elif self.generator_provider == "gemini":
            self.client = genai.Client(api_key=generator_config["api_key"])  # Force AI Studio API
        elif self.generator_provider == "groq":
            self.client = Groq(api_key=generator_config["api_key"])
        else:
            raise ValueError(f"Unsupported generator provider: {self.generator_provider}")
        
        # Initialize ROUGE scorer and threshold
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.rouge_cross_contradiction_threshold = 0.9
        
        self.contradiction_definition = """
A contradiction occurs when two statements make claims that cannot both be true simultaneously. Contradictions can be implicit as well as explicit.
Examples:
- "The paper is clearly written" vs "The presentation is overly confusing"
- "Results are credible" vs "Cannot fully trust the conclusions"
- "Experiments are satisfactory" vs "Need to increase scope of experiments"
"""
        
        # Define aspects to check
        self.aspects = {
            "Substance": "The paper lacks substantial experiments or detailed analyses (e.g., insufficient experiments, poor result analysis, missing ablation studies).",
            "Motivation": "The paper fails to address an important problem or its significance is questionable (e.g., the research lacks impact or relevance).",
            "Clarity": "The paper is poorly written, unorganized, or unclear about its contributions and methodology.",
            "Meaningful comparison": "The paper does not fairly compare its methods with prior work or omits necessary comparative analysis.",
            "Originality": "The paper does not offer new research topics, techniques, or insights, or its contributions are incremental.",
            "Soundness": "The paper's methodology or claims are not convincingly supported or are logically inconsistent."
        }
        
        # Initialize multi-agent debate scorer
        self.debate_scorer = MultiAgentDebateScorer(
            agent1_config=agent1_config,
            agent2_config=agent2_config,
            judge_config=judge_config
        )
    
    def _create_aspect_prompt(self, aspect_name: str, aspect_description: str, 
                             review_1: str, review_2: str) -> str:
        """Create prompt for a single aspect."""
        return f"""CONTRADICTION DEFINITION:
{self.contradiction_definition}

TASK:
Analyze these two peer reviews and identify both implicit as well as explicit contradictions specifically related to the "{aspect_name}" aspect.

ASPECT FOCUS: {aspect_name}
Description: {aspect_description}

REVIEW 1:
{review_1}

REVIEW 2:
{review_2}

INSTRUCTIONS:
1. Look for statements in both reviews that relate to "{aspect_name}"
2. Identify if these statements contradict each other
3. If contradictions exist, extract them with exact evidence from both reviews
4. Extract as many contradictions you can find related to "{aspect_name}"
4. It is NOT necessary that you find contradictions for this Aspect; if none exist, return an empty list

Output contradictions in this JSON format:
{{
  "aspect": "{aspect_name}",
  "contradictions": [
    {{
      "contradiction": "brief description of contradiction",
      "evidence": ["exact quote from Review 1", "exact quote from Review 2"]
    }}
  ]
}}

RULES:
- Focus ONLY on the "{aspect_name}" aspect
- Evidence array must have exactly 2 elements: [Review_1_quote, Review_2_quote]
- Extract the exact complete sentences from each review that illustrate the contradiction
- If no contradictions found for this aspect, return empty contradictions array
- Output ONLY valid JSON, no additional text"""
    
    def detect_aspect_contradictions(self, aspect_name: str, aspect_description: str,
                                    review_1: str, review_2: str) -> List[Dict]:
        """Detect contradictions for a single aspect."""
        prompt = self._create_aspect_prompt(aspect_name, aspect_description, review_1, review_2)
        
        try:
            if self.generator_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.generator_model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                
            elif self.generator_provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.generator_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=2048,
                    )
                )
                content = response.text
                
            elif self.generator_provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.generator_model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
            
            result = self._parse_json_response(content)
            
            contradictions = result.get("contradictions", [])
            for c in contradictions:
                c["aspect"] = aspect_name
            
            return contradictions
        
        except Exception as e:
            print(f"      Error on aspect '{aspect_name}': {str(e)}")
            return []
    
    def _calculate_rouge_l(self, sentence_a: str, sentence_b: str) -> float:
        """Calculates ROUGE-L F1 score between two sentences."""
        if not sentence_a or not sentence_b:
            return 0.0
        try:
            scores = self.rouge_scorer.score(sentence_a, sentence_b)
            return scores["rougeL"].fmeasure
        except Exception as e:
            print(f"        ROUGE calculation error: {str(e)}")
            return 0.0
    
    def _calculate_contradiction_similarity(self, evidence_1: List[str], evidence_2: List[str]) -> float:
        """Calculate similarity between two contradictions based on their evidence."""
        if len(evidence_1) != 2 or len(evidence_2) != 2:
            return 0.0
        
        rouge_review1 = self._calculate_rouge_l(evidence_1[0], evidence_2[0])
        rouge_review2 = self._calculate_rouge_l(evidence_1[1], evidence_2[1])
        
        return max(rouge_review1, rouge_review2)
    
    def detect_contradictions(self, paper_id: str, review_1: str, review_2: str) -> Dict:
        """
        Detect contradictions with multi-agent debate intensity scoring.
        Only keeps contradictions with intensity >= 1.
        Uses parallel processing for aspect contradiction generation.
        """
        all_contradictions = []
        contradiction_counter = 1
        
        # Phase 1: Generate contradictions per aspect IN PARALLEL
        print(f"    Phase 1: Generating contradictions (6 API calls in parallel)")
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Create futures for all aspects
            future_to_aspect = {
                executor.submit(
                    self.detect_aspect_contradictions,
                    aspect_name,
                    aspect_description,
                    review_1,
                    review_2
                ): aspect_name
                for aspect_name, aspect_description in self.aspects.items()
            }
            
            # Collect results as they complete
            from concurrent.futures import as_completed
            for future in as_completed(future_to_aspect):
                aspect_name = future_to_aspect[future]
                try:
                    aspect_contradictions = future.result()
                    print(f"      ✓ {aspect_name}: {len(aspect_contradictions)} contradictions")
                    all_contradictions.extend(aspect_contradictions)
                except Exception as e:
                    print(f"      ✗ {aspect_name}: Error - {str(e)}")
        
        print(f"      Total generated: {len(all_contradictions)} contradictions")
        
        # Phase 2: Structural validation and ROUGE-based de-duplication
        print(f"    Phase 2: De-duplication and filtering")
        
        structural_filtered = 0
        rouge_filtered_count = 0
        unique_contradictions = []
        
        for contradiction in all_contradictions:
            evidence = contradiction.get("evidence", [])
            
            # Structural Check
            if not (isinstance(evidence, list) and len(evidence) == 2 and 
                    evidence[0].strip() and evidence[1].strip()):
                structural_filtered += 1
                continue
            
            # Check ROUGE similarity
            is_duplicate = False
            for existing in unique_contradictions:
                existing_evidence = existing.get("evidence", [])
                similarity = self._calculate_contradiction_similarity(evidence, existing_evidence)
                
                if similarity >= self.rouge_cross_contradiction_threshold:
                    is_duplicate = True
                    rouge_filtered_count += 1
                    print(f"        Filtered (ROUGE > {self.rouge_cross_contradiction_threshold}): "
                          f"Similarity {similarity:.3f}")
                    break
            
            if not is_duplicate:
                unique_contradictions.append(contradiction)
        
        # Phase 3: Multi-Agent Debate Intensity Scoring
        print(f"    Phase 3: Multi-agent debate scoring ({len(unique_contradictions)} to score)")
        
        intensity_scored_contradictions = []
        zero_intensity_filtered = 0
        intensity_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        debate_stats = {"agreements": 0, "debates": 0, "judge_calls": 0,
    "judge_picked_agent1": 0,
    "judge_picked_agent2": 0,
    "judge_picked_neither": 0}
        
        for idx, contradiction in enumerate(unique_contradictions, 1):
            print(f"        Scoring contradiction {idx}/{len(unique_contradictions)}...")
            evidence = contradiction.get("evidence", [])
            
            # Multi-agent debate scoring
            debate_result = self.debate_scorer.score_with_debate(
                evidence, review_1, review_2
            )
            
            intensity = debate_result["intensity"]
            
            # Add debate metadata
            contradiction["intensity"] = intensity
            contradiction["intensity_reasoning"] = debate_result["reasoning"]
            contradiction["debate_history"] = debate_result["debate_history"]
            contradiction["debate_agreement"] = debate_result["agreement"]
            contradiction["debate_rounds"] = debate_result["rounds"]
            
            # Track statistics
            intensity_distribution[intensity] += 1
            
            if debate_result["agreement"]:
                debate_stats["agreements"] += 1
            else:
                debate_stats["debates"] += 1
                debate_stats["judge_calls"] += 1
            judge_pick = debate_result.get("judge_agreed_with", "Neither")
            if judge_pick == "Agent_1":
                debate_stats["judge_picked_agent1"] += 1
            elif judge_pick == "Agent_2":
                debate_stats["judge_picked_agent2"] += 1
            else:
                debate_stats["judge_picked_neither"] += 1
            # Only keep contradictions with intensity >= 1
            if intensity >= 1:
                contradiction["contradiction_id"] = f"C{contradiction_counter:03d}"
                contradiction["paper_id"] = paper_id
                contradiction_counter += 1
                intensity_scored_contradictions.append(contradiction)
            else:
                zero_intensity_filtered += 1
        
        # Statistics reporting
        print(f"    Phase 2-3 Statistics:")
        print(f"      Initial contradictions: {len(all_contradictions)}")
        print(f"      Structural filtered: {structural_filtered}")
        print(f"      ROUGE duplicate filtered: {rouge_filtered_count}")
        print(f"      Unique contradictions: {len(unique_contradictions)}")
        print(f"      Intensity distribution: {intensity_distribution}")
        print(f"      Zero-intensity filtered: {zero_intensity_filtered}")
        print(f"      Debate statistics:")
        print(f"        - Immediate agreements: {debate_stats['agreements']}")
        print(f"        - Required debates: {debate_stats['debates']}")
        print(f"        - Judge interventions: {debate_stats['judge_calls']}")
        print(f"          * Judge picked Agent_1: {debate_stats['judge_picked_agent1']}")
        print(f"          * Judge picked Agent_2: {debate_stats['judge_picked_agent2']}")
        print(f"          * Judge picked Neither: {debate_stats['judge_picked_neither']}")
        print(f"      Final contradictions (intensity ≥ 1): {len(intensity_scored_contradictions)}")
        print(f"      Total filtered: {len(all_contradictions) - len(intensity_scored_contradictions)}")
        
        return {
            "Review_1_full": review_1,
            "Review_2_full": review_2,
            "analysis": intensity_scored_contradictions,
            "intensity_distribution": intensity_distribution,
            "debate_statistics": debate_stats
        }
    
    def _parse_json_response(self, response: str) -> Dict:
        """Extract JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            return {"contradictions": []}


def load_ground_truth_data(file_path: str) -> Dict:
    """Load ground truth data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_all_papers(
    ground_truth_file: str,
    output_file: str,
    generator_config: Dict,
    agent1_config: Dict,
    agent2_config: Dict,
    judge_config: Dict,
    limit: Optional[int] = None
) -> Dict:
    """
    Process all papers with multi-agent debate system using multiple LLM providers.
    Supports incremental processing - resumes from last successful paper.
    
    Args:
        ground_truth_file: Path to input JSON file
        output_file: Path to output JSON file
        generator_config: Config for contradiction generator
        agent1_config: Config for first evaluator agent
        agent2_config: Config for second evaluator agent  
        judge_config: Config for judge agent
        limit: Optional limit on number of papers to process
    """
    all_data = load_ground_truth_data(ground_truth_file)
    
    # Load existing results to support resumption
    existing_results = load_existing_output(output_file)
    
    total_papers = len(all_data)
    if limit:
        total_papers = min(limit, total_papers)
    
    # Count how many are already done
    already_processed = sum(1 for paper_id in list(all_data.keys())[:total_papers] 
                           if paper_id in existing_results)
    
    print(f"Processing {total_papers} papers from {ground_truth_file}")
    print(f"Already processed: {already_processed} papers (will skip)")
    print(f"Remaining: {total_papers - already_processed} papers")
    print(f"Generator: {generator_config['provider']} - {generator_config['model']}")
    print(f"Evaluator Agent 1: {agent1_config['provider']} - {agent1_config['model']}")
    print(f"Evaluator Agent 2: {agent2_config['provider']} - {agent2_config['model']}")
    print(f"Judge: {judge_config['provider']} - {judge_config['model']}")
    print(f"Strategy: Multi-Agent Debate with 3-round maximum + Judge")
    print(f"Output: {output_file}\n")
    
    detector = ExternalIterativeDetector(
        generator_config=generator_config,
        agent1_config=agent1_config,
        agent2_config=agent2_config,
        judge_config=judge_config
    )
    
    # Track statistics for newly processed papers only
    newly_processed = 0
    total_final = 0
    total_intensity_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    total_debate_stats = {
        "agreements": 0, "debates": 0, "judge_calls": 0,
        "judge_picked_agent1": 0,
        "judge_picked_agent2": 0,
        "judge_picked_neither": 0
    }
    
    for i, (paper_id, paper_info) in enumerate(all_data.items()):
        if limit and i >= limit:
            break
        
        # Skip if already processed
        if paper_id in existing_results:
            print(f"\n[{i+1}/{total_papers}] Skipping {paper_id} (already processed)")
            continue
        
        print(f"\n[{i+1}/{total_papers}] Processing: {paper_id}")
        
        paper_data = all_data[paper_id]
        review_keys = [key for key in paper_data.keys() 
                      if key.startswith('Review_') and key.endswith('_full')]

        if len(review_keys) < 2:
            print(f"  Skipping {paper_id}: Not enough reviews")
            # Save empty result to mark as processed
            append_result(paper_id, {
                "error": "Not enough reviews",
                "Review_1_full": "",
                "Review_2_full": "",
                "analysis": [],
                "intensity_distribution": {0: 0, 1: 0, 2: 0, 3: 0},
                "debate_statistics": total_debate_stats.copy()
            }, output_file)
            continue

        review_1 = paper_data.get(review_keys[0], '')
        review_2 = paper_data.get(review_keys[1], '')
        
        try:
            result = detector.detect_contradictions(paper_id, review_1, review_2)
            
            # Immediately save result
            append_result(paper_id, result, output_file)
            
            # Update statistics
            newly_processed += 1
            num_final = len(result['analysis'])
            total_final += num_final
            
            for intensity, count in result.get('intensity_distribution', {}).items():
                total_intensity_dist[int(intensity)] += count
            
            for key in total_debate_stats.keys():
                total_debate_stats[key] += result.get('debate_statistics', {}).get(key, 0)
            
            print(f"    ✓ Saved: {num_final} contradictions with intensity ≥ 1")
            
        except Exception as e:
            print(f"    ✗ Error processing {paper_id}: {str(e)}")
            # Save error result to mark as attempted
            append_result(paper_id, {
                "error": str(e),
                "Review_1_full": review_1,
                "Review_2_full": review_2,
                "analysis": [],
                "intensity_distribution": {0: 0, 1: 0, 2: 0, 3: 0},
                "debate_statistics": total_debate_stats.copy()
            }, output_file)
    
    # Load final results for summary
    final_results = load_existing_output(output_file)
    
    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Total papers in output: {len(final_results)}")
    print(f"✓ Newly processed this run: {newly_processed}")
    print(f"✓ Total generator API calls (this run): {newly_processed * 6}")
    print(f"✓ New contradictions found (intensity ≥ 1): {total_final}")
    print(f"✓ Intensity distribution (this run):")
    for intensity in sorted(total_intensity_dist.keys()):
        print(f"    Intensity {intensity}: {total_intensity_dist[intensity]} contradictions")
    print(f"✓ Debate statistics (this run):")
    print(f"    Immediate agreements: {total_debate_stats['agreements']}")
    print(f"    Required debates: {total_debate_stats['debates']}")
    print(f"    Judge interventions: {total_debate_stats['judge_calls']}")
    print(f"      - Judge picked Agent_1: {total_debate_stats['judge_picked_agent1']}")
    print(f"      - Judge picked Agent_2: {total_debate_stats['judge_picked_agent2']}")
    print(f"      - Judge picked Neither: {total_debate_stats['judge_picked_neither']}")
    print(f"{'='*60}")
    
    return final_results


if __name__ == "__main__":
    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    # File paths
    GROUND_TRUTH_FILE = "ground_data.json"
    OUTPUT_FILE = "open_source_performance.json"
    
    # Configuration for each component with different providers
    generator_config = {
        "provider": "openai",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4o-mini" 
    }
    
    agent2_config = {
        "provider": "gemini",
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model": "gemini-3-flash-preview" 
    }
    
    agent1_config = {
        "provider": "groq",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct" 
    }
    
    judge_config = {
        "provider": "openai",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-5.1" 
    }
    
    # Process all papers
    results = process_all_papers(
        ground_truth_file=GROUND_TRUTH_FILE,
        output_file=OUTPUT_FILE,
        generator_config=generator_config,
        agent1_config=agent1_config,
        agent2_config=agent2_config,
        judge_config=judge_config,
        limit=None  # Set to a number to limit papers, or None for all
    )