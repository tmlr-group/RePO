"""Reward functions for GRPO training."""

import math
import os
import re
from typing import Dict
import requests
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from openai import OpenAI
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Initialize OpenAI client
client = OpenAI(
    api_key="",
    base_url=""
)

def extract_reference_from_prompt(prompt):
    """Extract reference SMILES string from prompt."""
    # Try to find SMILES after "molecule:" or "SMILES:"
    match = re.search(r"molecule:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)", prompt, re.IGNORECASE)
    if not match:
        match = re.search(r"SMILES?:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)", prompt, re.IGNORECASE)
    
    if match:
        smile = match.group(1).strip()
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                return smile
        except:
            pass
    
    # Try to find potential SMILES strings
    # Pre-filter: exclude common non-SMILES words
    common_words = ["think", "answer", "molecule", "modify", "optimize", "increase", "decrease", 
                    "value", "please", "smiles", "structure", "similar", "drug-like", "maintaining"]
    
    # More precise SMILES pattern requiring at least one element symbol
    smiles_pattern = r'([A-Za-z0-9@\[\]\(\)\\.=#\-\+]{5,})'
    matches = re.findall(smiles_pattern, prompt)
    
    for potential_smile in matches:
        # Skip common non-SMILES words
        if potential_smile.lower() in common_words:
            continue
            
        # Skip strings without any chemical element symbols
        if not re.search(r'[CNOPS]|Cl|Br|Fe|Si|Al|Ca|Mg|Na|Li|He|Ne|Ar|Kr|Xe|Rn', potential_smile):
            continue
            
        try:
            mol = Chem.MolFromSmiles(potential_smile)
            if mol is not None:
                return potential_smile
        except:
            continue
    
    return None

def infer_direction_from_prompt(prompt):
    """Infer optimization direction from prompt."""
    lower_prompt = prompt.lower()
    if "increase" in lower_prompt or "higher" in lower_prompt or "maximize" in lower_prompt:
        return "increase"
    elif "decrease" in lower_prompt or "lower" in lower_prompt or "minimize" in lower_prompt:
        return "decrease"
    return "increase"  # Fallback default

def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text

def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def evaluate_answer_similarity(answer, solution):
    """Use GPT4O-mini to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical answer evaluator. Compare the student's answer with the correct solution and output ONLY '1.0' if they match in meaning, or '0.0' if they don't match. No other output is allowed."
                },
                {
                    "role": "user",
                    "content": f"Student answer: {answer}\nCorrect solution: {solution}\nOutput only 1.0 or 0.0:"
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return float(result)
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if normalize_text(answer) == normalize_text(solution) else 0.0

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # First try latex parsing
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # print('latex gold parsed')
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
            # print('\nprompt:', prompt)
            print('-'*100)
            print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        else:
            # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
            answer_content = extract_answer(content)
            normalized_content = normalize_text(answer_content)
            normalized_solution = normalize_text(sol)
            reward = evaluate_answer_similarity(normalized_content, normalized_solution)
            print('-'*100)
            print('\nanswer_parsed:', normalized_content, '\ngold_parsed:', normalized_solution, '\nreward:', reward)
        rewards.append(reward)

    print('\naccuracy rewards:', rewards)

    return rewards


def accuracy_answer_reward(completion, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    '''
    input is completion string, answer is extracted gold answer.
    '''
    gold_parsed = answer
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        reward = float(verify(answer_parsed, gold_parsed))
        print('-'*100)
        print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
    return reward


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    print('-'*100)
    print('\nformat rewards:', rewards)
    return rewards


def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(prompts, completions, solutions=None, min_tokens=100, target_tokens=500, max_reward=0.8, reward_curve="linear", **kwargs):
    """
    Reward function that encourages longer responses up to a target length.
    
    Args:
        prompts: List of prompt strings (not used in this function)
        completions: List of completion strings or list of dicts with 'content' key
        solutions: List of solution strings (optional, not used in this function)
        min_tokens: Minimum number of tokens required to start receiving a reward
        target_tokens: Target number of tokens that will receive the maximum reward
        max_reward: Maximum reward value for reaching the target length
        reward_curve: How the reward scales between min and target lengths ("linear", "quadratic", "logarithmic")
        **kwargs: Additional keyword arguments that will be ignored
    
    Returns:
        List of reward values for each completion
    """
    rewards = []
    
    # Handle different completion input formats
    processed_completions = []
    for completion in completions:
        if isinstance(completion, dict) and "content" in completion:
            # If dict format, extract the content field
            processed_completions.append(completion["content"])
        elif isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) and "content" in completion[0]:
            # If grpo.py format: [{"content": "..."}]
            processed_completions.append(completion[0]["content"])
        else:
            # Otherwise assume it is already a raw string
            processed_completions.append(completion)
    
    for completion_text in processed_completions:
        # Simple tokenization by splitting on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', completion_text)
        token_count = len(tokens)
        
        # Print debugging info
        # print(f"Token count: {token_count}, min_tokens: {min_tokens}, target_tokens: {target_tokens}")
        
        # No reward if below minimum length
        if token_count < min_tokens:
            rewards.append(0.0)
            continue
            
        # Full reward if at or above target length
        if token_count >= target_tokens:
            rewards.append(max_reward)
            continue
            
        # Calculate normalized progress from min to target
        progress = (token_count - min_tokens) / (target_tokens - min_tokens)
        
        # Apply different reward curves
        if reward_curve == "linear":
            # Linear scaling
            reward = progress * max_reward
        elif reward_curve == "quadratic":
            # Quadratic scaling (rewards longer completions more)
            reward = (progress ** 2) * max_reward
        elif reward_curve == "logarithmic":
            # Logarithmic scaling (rewards initial progress more)
            # Avoid log(0)
            log_progress = math.log(1 + 9 * progress) / math.log(10)
            reward = log_progress * max_reward
        else:
            # Default to linear
            reward = progress * max_reward
            
        rewards.append(reward)
    
    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    # Check whether math_verify is available
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        math_verify_available = True
    except ImportError:
        math_verify_available = False
        print("Warning: math_verify module not available, using simple string matching instead")
        
        # Define lightweight fallback helpers
        def simple_parse(text, **kwargs):
            # Naively extract content between $...$ or $$...$$ as a math expression
            match = re.search(r'\$\$(.*?)\$\$|\$(.*?)\$', text, re.DOTALL)
            if match:
                expr = match.group(1) or match.group(2)
                return [expr.strip()]
            return []
        
        def simple_verify(answer, gold):
            # Compare two expressions directly (ignoring whitespace)
            if not answer or not gold:
                return False
            return answer[0].replace(' ', '') == gold[0].replace(' ', '')
        
        # Replace parser/verify with fallback versions
        parse = simple_parse
        verify = simple_verify
        LatexExtractionConfig = type('LatexExtractionConfig', (), {})
        NormalizationConfig = type('NormalizationConfig', (), {})

    def cosine_scaled_reward(completions, solution=None, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions (optional)

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        # If no solution is provided (or empty), use a length-based default reward
        if solution is None or len(solution) == 0 or all(not sol for sol in solution):
            for content in contents:
                gen_len = len(content)
                # Apply length-based cosine scaling
                progress = gen_len / max_len
                cosine = math.cos(progress * math.pi)
                # Use a neutral reward value (between correct and wrong ranges)
                neutral_min = (min_value_correct + max_value_wrong) / 2
                neutral_max = (max_value_correct + min_value_wrong) / 2
                reward = neutral_min + 0.5 * (neutral_max - neutral_min) * (1.0 + cosine)
                rewards.append(float(reward))
            return rewards

        # Standard reward computation logic
        for content, sol in zip(contents, solution):
            try:
                if math_verify_available:
                    gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
                    if len(gold_parsed) == 0:
                        # If the gold solution cannot be parsed, use a neutral reward
                        gen_len = len(content)
                        progress = gen_len / max_len
                        cosine = math.cos(progress * math.pi)
                        neutral_min = (min_value_correct + max_value_wrong) / 2
                        neutral_max = (max_value_correct + min_value_wrong) / 2
                        reward = neutral_min + 0.5 * (neutral_max - neutral_min) * (1.0 + cosine)
                        rewards.append(float(reward))
                        print("Failed to parse gold solution: ", sol)
                        continue

                    answer_parsed = parse(
                        content,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    equations=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )

                    is_correct = verify(answer_parsed, gold_parsed)
                else:
                    # Use simple string matching
                    gold_parsed = parse(sol)
                    answer_parsed = parse(content)
                    is_correct = verify(answer_parsed, gold_parsed)
                
                gen_len = len(content)

                # Apply cosine scaling based on length
                progress = gen_len / max_len
                cosine = math.cos(progress * math.pi)

                if is_correct:
                    min_value = min_value_correct
                    max_value = max_value_correct
                else:
                    # Swap min/max for incorrect answers
                    min_value = max_value_wrong
                    max_value = min_value_wrong

                reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
                rewards.append(float(reward))
            except Exception as e:
                # Handle any unexpected exception
                print(f"Error in cosine_scaled_reward: {e}")
                # Return a neutral reward
                gen_len = len(content)
                progress = gen_len / max_len
                cosine = math.cos(progress * math.pi)
                neutral_min = (min_value_correct + max_value_wrong) / 2
                neutral_max = (max_value_correct + min_value_wrong) / 2
                reward = neutral_min + 0.5 * (neutral_max - neutral_min) * (1.0 + cosine)
                rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def get_smile_validity_reward(
    validity_weight: float = 1.0,
    property_weights: dict = None,
    target_properties: dict = None,
    extract_pattern: str = r"<smile>(.*?)</smile>"
):
    """
    Reward function for evaluating the validity of generated SMILE strings and their properties.
    
    Args:
        validity_weight: Weight for the validity component of the reward
        property_weights: Dictionary mapping property names to their weights in the reward calculation
        target_properties: Dictionary mapping property names to their target values
        extract_pattern: Regex pattern to extract SMILE string from completion
        
    Returns:
        Reward function that evaluates SMILE strings
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        raise ImportError("RDKit is required for SMILE validity reward. Install with 'pip install rdkit'")
    
    # Default property weights if none provided
    if property_weights is None:
        property_weights = {}
    
    # Default target properties if none provided
    if target_properties is None:
        target_properties = {}
    
    def mol_prop(smile_str, prop):
        """Evaluate molecular property using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smile_str)
        except:
            return None
            
        # Check if molecule is valid
        if mol is None:
            return None
            
        # Basic molecular properties
        if prop == 'validity':
            return True
        elif prop == 'logP':
            return Descriptors.MolLogP(mol)
        elif prop == 'weight':
            return Descriptors.MolWt(mol)
        elif prop == 'qed':
            return Descriptors.qed(mol)
        elif prop == 'MR':
            return Descriptors.MolMR(mol)
        elif prop == 'TPSA':
            return Descriptors.TPSA(mol)
        elif prop == 'HBA':  # Hydrogen Bond Acceptor
            return Descriptors.NumHAcceptors(mol)
        elif prop == 'HBD':  # Hydrogen Bond Donor
            return Descriptors.NumHDonors(mol)
        elif prop == 'rot_bonds':  # Rotatable bonds
            return Descriptors.NumRotatableBonds(mol)
        elif prop == 'ring_count':
            return Descriptors.RingCount(mol)
        
        return None
    
    def extract_smile(text):
        """
        Extract and validate a SMILES string
        
        Args:
            text: Text to extract SMILES from
            
        Returns:
            Extracted SMILES string; return empty string if extraction/validation fails
        """
        if text is None:
            return ""
        
        # 1. First try extraction with the primary pattern (extract_pattern argument)
        match = re.search(extract_pattern, text, re.DOTALL)
        if match:
            smile = match.group(1).strip()
            # Validate SMILES
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass  # If validation fails, continue with fallback methods
        
        # 2. Try fallback pattern: <smile> tags
        match = re.search(r"<smile>(.*?)</smile>", text, re.DOTALL)
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # 3. Try extracting SMILES from text context
        # Look for text after "SMILES:" or "SMILE:"
        match = re.search(r"SMILES?:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)", text, re.IGNORECASE)
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # 4. Try heuristic extraction for possible SMILES strings
        # Pre-filter: exclude common non-SMILES words
        common_words = ["think", "answer", "molecule", "modify", "optimize", "increase", "decrease", 
                        "value", "please", "smiles", "structure", "similar", "drug-like", "maintaining"]
        
        # More precise SMILES pattern requiring at least one element symboland structural features
        smiles_pattern = r'([A-Za-z0-9@\[\]\(\)\\.=#\-\+]{5,})'
        matches = re.findall(smiles_pattern, text)
        
        for potential_smile in matches:
            # Skip common non-SMILES words
            if potential_smile.lower() in common_words:
                continue
                
            # Skip strings without any chemical element symbols
            if not re.search(r'[CNOPS]|Cl|Br|Fe|Si|Al|Ca|Mg|Na|Li|He|Ne|Ar|Kr|Xe|Rn', potential_smile):
                continue
                
            try:
                mol = Chem.MolFromSmiles(potential_smile)
                if mol is not None and Chem.MolToSmiles(mol) == potential_smile:
                    return potential_smile
            except:
                continue
        
        return ""  # Return an empty string if all methods fail
    
    def smile_validity_reward(completions, **kwargs) -> float:
        """
        Reward function that evaluates the validity of generated SMILE strings and their properties.
        
        Args:
            completions: List of model completions
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for completion in contents:
            # Extract SMILE string from completion
            smile_str = extract_smile(completion)
            
            # Initialize reward components
            validity_reward = 0.0
            property_rewards = {}
            
            # Check validity
            if mol_prop(smile_str, "validity") is not None:
                validity_reward = 1.0
                
                # Calculate property rewards if molecule is valid
                for prop, weight in property_weights.items():
                    prop_value = mol_prop(smile_str, prop)
                    
                    if prop_value is not None and prop in target_properties:
                        target = target_properties[prop]
                        # Normalize the difference between actual and target values
                        # The closer to the target, the higher the reward
                        diff = abs(prop_value - target)
                        max_diff = max(abs(target), 1.0)  # Avoid division by zero
                        property_rewards[prop] = weight * (1.0 - min(diff / max_diff, 1.0))
            
            # Calculate total reward
            total_reward = validity_weight * validity_reward
            for prop_reward in property_rewards.values():
                total_reward += prop_reward
                
            # Normalize total reward to [0, 1] range
            total_weight = validity_weight + sum(property_weights.values())
            if total_weight > 0:
                total_reward /= total_weight
                
            rewards.append(total_reward)
            
            # Print debugging information
            print('-'*100)
            print(f'SMILE: {smile_str}')
            print(f'Validity: {validity_reward}')
            print(f'Property rewards: {property_rewards}')
            print(f'Total reward: {total_reward}')
            
        return rewards
    
    return smile_validity_reward


def get_smile_similarity_reward(
    reference_smiles: list = None,
    extract_pattern: str = r"<smile>(.*?)</smile>",
    similarity_threshold: float = 0.0,
    similarity_target: float = None,
    n_bits: int = 2048,
    reward_mode: str = "max"
):
    """
    Reward function for evaluating the similarity between generated molecules and reference molecules.
    
    Args:
        reference_smiles: List of reference SMILE strings to compare against
        extract_pattern: Regex pattern to extract SMILE string from completion
        similarity_threshold: Minimum similarity threshold for a valid reward
        similarity_target: Target similarity value (if None, higher similarity is better)
        n_bits: Number of bits for Morgan fingerprint
        reward_mode: How to aggregate similarities if multiple references ("max", "min", "mean")
        
    Returns:
        Reward function that evaluates molecular similarity
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit is required for SMILE similarity reward. Install with 'pip install rdkit'")
    
    if reference_smiles is None:
        reference_smiles = []
    
    # Pre-compute reference fingerprints
    reference_fps = []
    morgan_gen = GetMorganGenerator(radius=2, fpSize=n_bits)
    for smile in reference_smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)
            reference_fps.append(fp)
    
    def extract_smile(text):
        """Extract SMILE string from text using regex pattern."""
        if text is None:
            return ""
        match = re.search(extract_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def calculate_similarity(smile_str):
        """Calculate similarity between a SMILE string and reference molecules."""
        try:
            mol = Chem.MolFromSmiles(smile_str)
            if mol is None:
                return 0.0
            
            # Generate fingerprint for the molecule
            fp = morgan_gen.GetFingerprint(mol)
            
            # Calculate similarities to all reference molecules
            similarities = []
            for ref_fp in reference_fps:
                similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
                similarities.append(similarity)
            
            # Return aggregated similarity based on mode
            if not similarities:
                return 0.0
            
            if reward_mode == "max":
                return max(similarities)
            elif reward_mode == "min":
                return min(similarities)
            else:  # Default to mean
                return sum(similarities) / len(similarities)
        except:
            return 0.0
    
    def smile_similarity_reward(completions, **kwargs) -> float:
        """
        Reward function that evaluates the similarity between generated molecules and reference molecules.
        
        Args:
            completions: List of model completions
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for completion in contents:
            # Extract SMILE string from completion
            smile_str = extract_smile(completion)
            
            # Calculate similarity
            similarity = calculate_similarity(smile_str)
            
            # Apply threshold
            if similarity < similarity_threshold:
                reward = 0.0
            else:
                # If target similarity is specified, reward closeness to target
                if similarity_target is not None:
                    # The closer to the target, the higher the reward
                    diff = abs(similarity - similarity_target)
                    reward = max(0.0, 1.0 - min(diff, 1.0))
                else:
                    # Otherwise, higher similarity is better
                    reward = similarity
            
            rewards.append(reward)
            
            # Print debugging information
            print('-'*100)
            print(f'SMILE: {smile_str}')
            print(f'Similarity: {similarity}')
            print(f'Reward: {reward}')
            
        return rewards
    
    return smile_similarity_reward


def get_smile_optimization_reward(
    property_name: str = "logP",
    target_direction: str = None,
    reference_smiles: str = None,
    similarity_weight: float = 0.5,
    property_weight: float = 0.5,
    min_similarity: float = 0.2,  # Lower similarity threshold to make non-zero reward easier to obtain
    extract_pattern: str = r"<answer>(.*?)</answer>"
):
    """
    Reward function for molecule optimization tasks, balancing property improvement and similarity.
    
    Args:
        property_name: Name of the property to optimize (e.g., 'logP', 'qed', 'MR')
        target_direction: Direction of optimization ('increase' or 'decrease'). If None, will be inferred from prompt.
        reference_smiles: Reference SMILE string to optimize. If None, will be extracted from prompt.
        similarity_weight: Weight for similarity component in the reward
        property_weight: Weight for property improvement component in the reward
        min_similarity: Minimum similarity threshold for a valid reward
        extract_pattern: Regex pattern to extract SMILE string from completion
        
    Returns:
        Reward function that evaluates molecular optimization
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:
        raise ImportError("RDKit is required for SMILE optimization reward. Install with 'pip install rdkit'")
    
    # Normalize property name to lowercase for case-insensitive matching
    property_name_lower = property_name.lower()
    
    def extract_smile(text):
        """
        Extract and validate a SMILES string
        
        Args:
            text: Text to extract SMILES from
            
        Returns:
            Extracted SMILES string; return empty string if extraction/validation fails
        """
        if text is None:
            return ""
        
        # 1. First try extraction with the primary pattern (extract_pattern argument)
        match = re.search(extract_pattern, text, re.DOTALL)
        if match:
            smile = match.group(1).strip()
            # Validate SMILES
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass  # If validation fails, continue with fallback methods
        
        # 2. Try fallback pattern: <smile> tags
        match = re.search(r"<smile>(.*?)</smile>", text, re.DOTALL)
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # 3. Try extracting SMILES from text context
        # Look for text after "SMILES:" or "SMILE:"
        match = re.search(r"SMILES?:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)", text, re.IGNORECASE)
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # 4. Try heuristic extraction for possible SMILES strings
        # Pre-filter: exclude common non-SMILES words
        common_words = ["think", "answer", "molecule", "modify", "optimize", "increase", "decrease", 
                        "value", "please", "smiles", "structure", "similar", "drug-like", "maintaining"]
        
        # More precise SMILES pattern requiring at least one element symboland structural features
        smiles_pattern = r'([A-Za-z0-9@\[\]\(\)\\.=#\-\+]{5,})'
        matches = re.findall(smiles_pattern, text)
        
        for potential_smile in matches:
            # Skip common non-SMILES words
            if potential_smile.lower() in common_words:
                continue
                
            # Skip strings without any chemical element symbols
            if not re.search(r'[CNOPS]|Cl|Br|Fe|Si|Al|Ca|Mg|Na|Li|He|Ne|Ar|Kr|Xe|Rn', potential_smile):
                continue
                
            try:
                mol = Chem.MolFromSmiles(potential_smile)
                if mol is not None and Chem.MolToSmiles(mol) == potential_smile:
                    return potential_smile
            except:
                continue
        
        return ""  # Return an empty string if all methods fail
    
    def extract_reference_from_prompt(prompt):
        """Extract reference SMILES string from prompt."""
        # Try to find SMILES after "molecule:" or "SMILES:"
        match = re.search(r"molecule:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)", prompt, re.IGNORECASE)
        if not match:
            match = re.search(r"SMILES?:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)", prompt, re.IGNORECASE)
        
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # Try to find potential SMILES strings
        # Pre-filter: exclude common non-SMILES words
        common_words = ["think", "answer", "molecule", "modify", "optimize", "increase", "decrease", 
                        "value", "please", "smiles", "structure", "similar", "drug-like", "maintaining"]
        
        # More precise SMILES pattern requiring at least one element symbol
        smiles_pattern = r'([A-Za-z0-9@\[\]\(\)\\.=#\-\+]{5,})'
        matches = re.findall(smiles_pattern, prompt)
        
        for potential_smile in matches:
            # Skip common non-SMILES words
            if potential_smile.lower() in common_words:
                continue
                
            # Skip strings without any chemical element symbols
            if not re.search(r'[CNOPS]|Cl|Br|Fe|Si|Al|Ca|Mg|Na|Li|He|Ne|Ar|Kr|Xe|Rn', potential_smile):
                continue
                
            try:
                mol = Chem.MolFromSmiles(potential_smile)
                if mol is not None:
                    return potential_smile
            except:
                continue
        
        return None
    
    def infer_direction_from_prompt(prompt):
        """Infer optimization direction from prompt."""
        lower_prompt = prompt.lower()
        if "increase" in lower_prompt or "higher" in lower_prompt or "maximize" in lower_prompt:
            return "increase"
        elif "decrease" in lower_prompt or "lower" in lower_prompt or "minimize" in lower_prompt:
            return "decrease"
        return "increase"  # Default to increase
    
    def calculate_property_improvement(mol, property_value, ref_property_value, direction):
        """Compute property improvement score."""
        if direction == "increase":
            if property_value > ref_property_value:
                return 1.0  # Successfully increased the property value
            else:
                return 0.0  # Failed to increase the property value
        else:  # decrease
            if property_value < ref_property_value:
                return 1.0  # Successfully decreased the property value
            else:
                return 0.0  # Failed to decrease the property value
    
    def smile_optimization_reward(completions, prompts=None, **kwargs) -> float:
        """
        Reward function that evaluates molecular optimization.
        
        Args:
            completions: List of model completions
            prompts: List of prompts (used to extract reference SMILES if not provided)
        """
        # Ensure there is content to process
        if isinstance(completions[0], list) and isinstance(completions[0][0], dict):
            contents = [completion[0]["content"] for completion in completions]
        elif isinstance(completions[0], dict) and "content" in completions[0]:
            contents = [completion["content"] for completion in completions]
        else:
            contents = completions
        
        rewards = []
        
        # If needed, extract reference_smiles and direction from prompt
        local_reference_smiles = reference_smiles
        local_target_direction = target_direction
        
        if (local_reference_smiles is None or local_target_direction is None) and prompts is not None:
            # Get the first prompt content
            if isinstance(prompts[0], list) and len(prompts[0]) > 0:
                prompt_text = prompts[0][-1]["content"] if isinstance(prompts[0][-1], dict) else prompts[0][-1]
            else:
                prompt_text = prompts[0]
            
            # Extract reference_smiles
            if local_reference_smiles is None:
                local_reference_smiles = extract_reference_from_prompt(prompt_text)
                if local_reference_smiles is None:
                    # Use default values if extraction from prompt fails
                    local_reference_smiles = "CC1CC(N2CCN(CC3CC3)CC2)C1"
            
            # Infer optimization direction
            if local_target_direction is None:
                local_target_direction = infer_direction_from_prompt(prompt_text)
        
        # Validate reference molecule
        ref_mol = Chem.MolFromSmiles(local_reference_smiles)
        if ref_mol is None:
            raise ValueError(f"Invalid reference SMILE string: {local_reference_smiles}")
        
        # Get reference property value
        ref_property_value = None
        if property_name_lower == 'logp':
            ref_property_value = Descriptors.MolLogP(ref_mol)
        elif property_name_lower == 'qed':
            ref_property_value = Descriptors.qed(ref_mol)
        elif property_name_lower == 'mr':
            ref_property_value = Descriptors.MolMR(ref_mol)
        elif property_name_lower == 'tpsa':
            ref_property_value = Descriptors.TPSA(ref_mol)
        else:
            raise ValueError(f"Unsupported property: {property_name}. Supported properties are: logP, QED, MR, TPSA")
        
        # Build reference fingerprint
        morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
        ref_fp = morgan_gen.GetFingerprint(ref_mol)
        
        for completion in contents:
            # Extract SMILE string
            smile_str = extract_smile(completion)
            
            try:
                mol = Chem.MolFromSmiles(smile_str)
            except:
                mol = None
            
            if mol is None:
                rewards.append(0.0)
                print('-'*100)
                print(f'SMILE: {smile_str}')
                print(f'Invalid molecule')
                continue
            
            # Compute molecular similarity
            fp = morgan_gen.GetFingerprint(mol)
            similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
            
            # If similarity is below threshold, assign zero reward
            if similarity < min_similarity:
                rewards.append(0.0)
                print('-'*100)
                print(f'SMILE: {smile_str}')
                print(f'Reference: {local_reference_smiles}')
                print(f'Similarity: {similarity} (below threshold {min_similarity})')
                print(f'Reward: 0.0')
                continue
            
            # Compute property value
            property_value = None
            if property_name_lower == 'logp':
                property_value = Descriptors.MolLogP(mol)
            elif property_name_lower == 'qed':
                property_value = Descriptors.qed(mol)
            elif property_name_lower == 'mr':
                property_value = Descriptors.MolMR(mol)
            elif property_name_lower == 'tpsa':
                property_value = Descriptors.TPSA(mol)
            
            # Compute property-improvement score (same logic as evaluate.py)
            improvement_score = calculate_property_improvement(
                mol, property_value, ref_property_value, local_target_direction
            )
            
            # Compute total reward as weighted sum of similarity and improvement
            reward = (similarity_weight * similarity) + (property_weight * improvement_score)
            
            rewards.append(reward)
            
            # Print debugging info
            print('-'*100)
            print(f'SMILE: {smile_str}')
            print(f'Reference: {local_reference_smiles}')
            print(f'Similarity: {similarity}')
            print(f'Property: {property_name}')
            print(f'Reference value: {ref_property_value}')
            print(f'Generated value: {property_value}')
            print(f'Direction: {local_target_direction}')
            print(f'Improvement score: {improvement_score}')
            print(f'Total reward: {reward}')
            
        return rewards
    
    return smile_optimization_reward



def get_multi_prop_optimization_reward(
    task: str = "bbbp+drd2+plogp",
    min_similarity: float = 0.5,
    extract_pattern: str = r"<answer>(.*?)</answer>"
):
    """
    Reward function for molecule optimization tasks, balancing property improvement and similarity.
    
    Args:
        property_name: Name of the property to optimize (e.g., 'logP', 'qed', 'MR')
        target_direction: Direction of optimization ('increase' or 'decrease'). If None, will be inferred from prompt.
        reference_smiles: Reference SMILE string to optimize. If None, will be extracted from prompt.
        similarity_weight: Weight for similarity component in the reward
        property_weight: Weight for property improvement component in the reward
        min_similarity: Minimum similarity threshold for a valid reward
        extract_pattern: Regex pattern to extract SMILE string from completion
        
    Returns:
        Reward function that evaluates molecular optimization
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:
        raise ImportError("RDKit is required for SMILE optimization reward. Install with 'pip install rdkit'")
    
    # NOTE: extract smiles from response
    def extract_smile(text):
        """
        Extract and validate a SMILES string
        
        Args:
            text: Text to extract SMILES from
            
        Returns:
            Extracted SMILES string; return empty string if extraction/validation fails
        """
        if text is None:
            return ""
        
        # 1. First try extraction with the primary pattern (extract_pattern argument)
        match = re.search(extract_pattern, text, re.DOTALL)
        if match:
            smile = match.group(1).strip()
            # Validate SMILES
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass  # If validation fails, continue with fallback methods
        
        # 2. Try fallback pattern: <smile> tags
        match = re.search(r"<smile>(.*?)</smile>", text, re.DOTALL)
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # 3. Try extracting SMILES from text context
        # Look for text after "SMILES:" or "SMILE:"
        match = re.search(r"SMILES?:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)", text, re.IGNORECASE)
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # 4. Try heuristic extraction for possible SMILES strings
        # Pre-filter: exclude common non-SMILES words
        common_words = ["think", "answer", "molecule", "modify", "optimize", "increase", "decrease", 
                        "value", "please", "smiles", "structure", "similar", "drug-like", "maintaining"]
        
        # More precise SMILES pattern requiring at least one element symboland structural features
        smiles_pattern = r'([A-Za-z0-9@\[\]\(\)\\.=#\-\+]{5,})'
        matches = re.findall(smiles_pattern, text)
        
        for potential_smile in matches:
            # Skip common non-SMILES words
            if potential_smile.lower() in common_words:
                continue
                
            # Skip strings without any chemical element symbols
            if not re.search(r'[CNOPS]|Cl|Br|Fe|Si|Al|Ca|Mg|Na|Li|He|Ne|Ar|Kr|Xe|Rn', potential_smile):
                continue
                
            try:
                mol = Chem.MolFromSmiles(potential_smile)
                if mol is not None and Chem.MolToSmiles(mol) == potential_smile:
                    return potential_smile
            except:
                continue
        
        return ""  # Return an empty string if all methods fail
    
    def extract_smiles_from_prompt(prompt):
        """Extract SMILES string from prompt."""
        match = re.search(r"Input : <SMILES>(.+?)</SMILES>", prompt, re.IGNORECASE)

        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                return ""
    
    def get_smiles_properties(smiles, require_drd2=True):
        # Define the base URLs of your running FastAPI servers
        ADMET_API_URL = "http://127.0.0.1:10086"  # ADMET Model API running on port 10086
        DRD2_API_URL = "http://127.0.0.1:10087"   # DRD2 Model API running on port 10087

        try:
            # Get ADMET results (just use POST for combined results)
            admet_response = requests.post(f"{ADMET_API_URL}/predict/", json={"smiles": smiles})
            admet_response.raise_for_status()
            combined_results = admet_response.json()
            
            if require_drd2:
                # Get DRD2 results (just use POST for combined results)
                drd2_response = requests.post(f"{DRD2_API_URL}/predict/", json={"smiles": smiles})
                drd2_response.raise_for_status()
                combined_results.update(drd2_response.json())
            
            return combined_results
        except requests.exceptions.RequestException as e:
            print(f"Error getting combined results: {e}")
            return {"mutagenicity": 0,"bbbp": 0,"qed": 0,"plogp": 0,"drd2": 0}

    def calculate_property_improvement(property_values, ref_property_values, task_required_properties):
        # {"bbbp": 0,"qed": 0,"plogp": 0,"drd2": 0}
        improvement = 0
        for prop_name in task_required_properties:
            response_prop_value = property_values[prop_name]
            ref_prop_value = ref_property_values[prop_name]
            # NOTE: all properties are to be increased
            if response_prop_value > ref_prop_value:
                improvement += 1
            else:
                improvement += 0
        return improvement / len(task_required_properties)

    def smile_multi_properties_optimization_reward(completions, prompts=None, **kwargs) -> float:
        """
        Reward function that evaluates molecular optimization.
        
        Args:
            completions: List of model completions
            prompts: List of prompts (used to extract reference SMILES if not provided)
        """
        # Ensure there is content to process
        if isinstance(completions[0], list) and isinstance(completions[0][0], dict):
            contents = [completion[0]["content"] for completion in completions]
        elif isinstance(completions[0], dict) and "content" in completions[0]:
            contents = [completion["content"] for completion in completions]
        else:
            contents = completions
        
        task_required_properties = task.split("+")
        rewards = []
        
        if prompts is not None:
            # Get the first prompt content
            if isinstance(prompts[0], list) and len(prompts[0]) > 0:
                prompt_text = prompts[0][-1]["content"] if isinstance(prompts[0][-1], dict) else prompts[0][-1]
            else:
                prompt_text = prompts[0]
            
            # Extract reference_smiles
            local_reference_smiles = extract_smiles_from_prompt(prompt_text)
        
        # Validate reference molecule
        ref_mol = Chem.MolFromSmiles(local_reference_smiles)
        if ref_mol is None:
            raise ValueError(f"Invalid reference SMILE string: {local_reference_smiles}")
        
        # Get reference property value
        if "drd2" in task:
            # {"bbbp": 0,"qed": 0,"plogp": 0,"drd2": 0}
            ref_property_values = get_smiles_properties(local_reference_smiles, require_drd2=True)
        else:
            # {"bbbp": 0,"qed": 0,"plogp": 0}
            ref_property_values = get_smiles_properties(local_reference_smiles, require_drd2=False)
        
        # Build reference fingerprint
        morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
        ref_fp = morgan_gen.GetFingerprint(ref_mol)
        
        for completion in contents:
            # Extract SMILE string
            smile_str = extract_smile(completion)
            
            try:
                mol = Chem.MolFromSmiles(smile_str)
            except:
                mol = None
            
            if mol is None:
                rewards.append(0.0)
                print('-'*100)
                print(f'SMILE: {smile_str}')
                print(f'Invalid molecule')
                continue
            
            # Compute molecular similarity
            fp = morgan_gen.GetFingerprint(mol)
            similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
            
            # If similarity is below threshold, assign zero reward
            if similarity < min_similarity:
                rewards.append(0.0)
                print('-'*100)
                print(f'SMILE: {smile_str}')
                print(f'Reference: {local_reference_smiles}')
                print(f'Similarity: {similarity} (below threshold {min_similarity})')
                print(f'Reward: 0.0')
                continue
            
            # Compute property value
            if "drd2" in task:
                # {"bbbp": 0,"qed": 0,"plogp": 0,"drd2": 0}
                response_property_values = get_smiles_properties(smile_str, require_drd2=True)
            else:
                # {"bbbp": 0,"qed": 0,"plogp": 0}
                response_property_values = get_smiles_properties(smile_str, require_drd2=False)
            
            # Compute property-improvement score (same logic as evaluate.py)
            improvement_score = calculate_property_improvement(response_property_values, ref_property_values, task_required_properties)
            
            # Compute total reward as weighted sum of similarity and improvement
            reward = 0.5 * similarity + 0.5 * improvement_score
            
            rewards.append(reward)
            
            # Print debugging info
            print('-'*100)
            print(f'SMILE: {smile_str}')
            print(f'Reference: {local_reference_smiles}')
            print(f'Similarity: {similarity}')
            print(f'Property: {task_required_properties}')
            print(f'Reference value: {ref_property_values}')
            print(f'Generated value: {response_property_values}')
            print(f'Direction: Increase')
            print(f'Improvement score: {improvement_score}')
            print(f'Total reward: {reward}')
            
        return rewards
    
    return smile_multi_properties_optimization_reward
