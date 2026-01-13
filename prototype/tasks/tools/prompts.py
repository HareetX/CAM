from langchain_core.prompts import PromptTemplate


ENTITY_EXTRACTION = """You are an advanced expert tasked with extracting entity concepts from a given text passage. For the input text provided below, identify the most dominant entity concepts mentioned (up to a maximum of FIVE), including specific names of people, items, and locations.

Please ensure the following:
- Avoid vague or generic descriptions (e.g., do not include 'a person' or 'some items').
- Eliminate any duplicates, i.e., each entity should appear only once.
- Focus solely on entities that are central or significant to the context, ignoring minor or incidental ones.
- Limit the total number of extracted entities to a maximum of FIVE, prioritizing the most critical ones.
- Present the result as a single line, with entities separated by semicolons (e.g., 'Entity1; Entity2; Entity3').

Input Passage:
{input_chunk}

Now, extract the dominant entity concepts, ensuring they are separated by semicolons:
"""


TEXT_SUMMARIZATION = """You are an advanced reading agent designed to autonomously summarize given narrative texts.

For the following series of narrative passages provided, your task is to generate a concise, coherent, and comprehensive summary.

Please ensure the following:
- Output your summary directly, keeping it moderately long, maintaining the narrative order, including essential details, but avoiding excessive length.
- Do not provide any additional explanations beyond the narrative summary.

**Narrative Passages:**
{input_texts}

**Summary:** """

KG_COMMUNITY_SUMMARIZATION = """You are an advanced knowledge graph agent designed to synthesize structured data into coherent narratives.

For the following set of Entities and Relationships representing a specific community within a knowledge graph, your task is to generate a concise, coherent, and comprehensive summary.

Please ensure the following:
- Synthesize the isolated facts (entities and relationships) into a fluid, natural language paragraph.
- Identify the central theme or event connecting the entities.
- Integrate all key details (who, what, where, actions) without simply listing the data rows.
- Ignore technical identifiers (IDs) and focus on the semantic meaning.
- Output your summary directly, keeping it moderately long, maintaining the narrative order, including essential details, but avoiding excessive length.
- Do not provide any additional explanations beyond the narrative summary.

**Graph Community Data:**
{input_texts}

**Summary:** """


MC_RESPONSE = """You are an advanced reading agent tasked with answering a question based on available passages. Given a input question, your goal is to select the correct answer from four candidate options (A, B, C, D) based on a set of passages. You should output your answer as a single letter: A, B, C, or D.

Please ensure the following:
- Analyze the input question and the given passages to determine the most accurate answer.
- Evaluate the four candidate options (A, B, C, D) and select the one that best answers the input question based on the provided information.
- Output your answer as a single letter: A, B, C, or D.
- Do not provide explanations or additional text—your response must be exactly one letter corresponding to the chosen option.

Here is the information you need to process:

**Input Question:** {question}

**Candidate Options:**
{options}

**Passages:**
{passages}

**Your Answer (A, B, C, or D):** """


GE_RESPONSE = """You are an advanced reading agent tasked with answering a question based on available passages. Given an input question, your goal is to provide a direct and accurate answer based on a set of passages.

Please ensure the following:
- Analyze the input question and the given passages to determine the answer.
- Provide a concise, direct answer to the question based solely on the information in the passages.
- Do not include any explanations or additional text beyond the answer itself. Keep your answer as concise and brief as possible.

Here is the information you need to process:

**Input Question:** {question}

**Passages:**
{passages}

**Your Answer:** """


CV_RESPONSE = """You are an advanced reading agent specialized in claim verification. Your task is to carefully read the provided Claim and the Context Passages, and determine whether the Claim is fully supported by the information in the Passages.

Please ensure the following:
- Analyze the Claim and the Passages carefully to decide your final judgment.
- Answer YES only if the Passages support the Claim; otherwise answer NO.
- Output exactly one word: YES or NO. Do not add any explanation or additional text.

**Claim:** {question}

**Passages:**
{passages}

**Your Answer:** """


SUM_RESPONSE = """You are an advanced reading agent specialized in query-based summarization. Your task is to generate a comprehensive and informative answer to the given Question based on the provided Context Passages.

Please ensure the following:
- Carefully analyze the Question to understand what information is being asked.
- Read through the Passages and synthesize relevant information to address the Question.
- Your answer should be comprehensive and focused on the Question, covering all key points supported by the Passages.

**Question:** {question}

**Passages:**
{passages}

**Your Answer:** """


JUDGE_PROMPT = """After reading some text, John was given the following question about the text:
{question}

John's answer to the question was:
{prediction}

The ground truth answer was:
{answer}

Does John's answer agree with the ground truth answer? (Please only answer YES or NO)
"""


PASSAGE_SELECTION = """You are an advanced reading agent skilled at identifying narrative text passages that are helpful for answering given questions. Given an input question and a series of narrative passages, your task is to select all passages that may be helpful for answering the given question.

The format of the provided passages is as follows:
Passage 1: Passage Text
Passage 2: Passage Text
Passage 3: Passage Text
...

Your should output a LIST containing all helpful passage numbers, e.g., [1, 3, 5].

**Question:** {input_question}

**Passages:**
{passages}

**A LIST containing all helpful passage numbers:** """

GIST_GENERATION = """Your task is to generate a concise gist that accurately summarizes the key information from the input Chunk.

Please ensure the following:
- Keep the gist brief, clear, and faithful to the original content.
- Do not include any additional explanation beyond what is in the Chunk.

**Input Chunk:** {input_chunk}

**Gist:** """

STATEMENT_EXTRACTION = """Your task is to extract a list of clear and concise statements that accurately reflect the key information from the input Chunk.

Please ensure the following:
    - Distill the input into normalized, atomic statements.
    - Keep the statements brief, clear, and faithful to the original content.

Return ONLY valid JSON strictly following this schema:
{{
  "statements": ["statement1", ...]
}}

**Input Chunk:** {input_chunk}

**Extracted Statements in JSON:** """

KG_EXTRACTION = """Your task is to extract a list of entities and their relationships from the input Chunk. Given a text chunk, first identify all entities needed from the text in order to capture the information and ideas in the text.
Next, report all relationships among the identified entities.

**Steps:**
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Suggest several labels or categories for the entity. The categories should not be specific, but should be as general as possible.
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as {{"graph_element": "entity", "entity_name": <entity_name>, "entity_type": <entity_type>, "entity_description": <entity_description>}}

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity (on a scale of 1 to 10, where 1 means a very weak relationship and 10 means a very strong relationship)
Format each relationship as {{"graph_element": "relationship", "source_entity": <source_entity>, "target_entity": <target_entity>, "relationship_description": <relationship_description>, "relationship_strength": <relationship_strength>}}

3. Return output in English as a JSON format Dict of all the entities and relationships identified in steps 1 and 2.
Format the output as a JSON format:
{{
  "graph_elements": {{
    "entities": [
      {{graph_element": "entity", "entity_name": <entity_name>, "entity_type": <entity_type>, "entity_description": <entity_description>}},
      ...
    ],
    "relationships": [
      {{graph_element": "relationship", "source_entity": <source_entity>, "target_entity": <target_entity>, "relationship_description": <relationship_description>, "relationship_strength": <relationship_strength>}},
      ...
  }}
}}

**Input Chunk:** {input_chunk}

**Extracted Graph Elements in JSON:**
"""


gist_generation_template = PromptTemplate(
                        input_variables=["input_chunk"],
                        template = GIST_GENERATION,
                        )

statement_extraction_template = PromptTemplate(
                        input_variables=["input_chunk"],
                        template = STATEMENT_EXTRACTION,
                        )

kg_extraction_template = PromptTemplate(
                        input_variables=["input_chunk"],
                        template=KG_EXTRACTION,
                        )

passage_selection_template = PromptTemplate(
                        input_variables=["input_question", "passages"],
                        template = PASSAGE_SELECTION,
                        )

entity_extraction_template = PromptTemplate(
                        input_variables=["input_chunk"],
                        template = ENTITY_EXTRACTION,
                        )

text_summarization_template = PromptTemplate(
                        input_variables=["input_texts"],
                        template = TEXT_SUMMARIZATION,
                        )

kg_community_summarization_template = PromptTemplate(
                        input_variables=["input_texts"],
                        template = KG_COMMUNITY_SUMMARIZATION,
                        )

final_response_mc_template = PromptTemplate(
                        input_variables=["question", "options", "passages"],
                        template = MC_RESPONSE,
                        )

final_response_ge_template = PromptTemplate(
                        input_variables=["question", "passages"],
                        template = GE_RESPONSE,
                        )

final_response_cv_template = PromptTemplate(
                        input_variables=["question", "passages"],
                        template = CV_RESPONSE,
                        )

final_response_sum_template = PromptTemplate(
                        input_variables=["question", "passages"],
                        template = SUM_RESPONSE,
                        )

judge_template = PromptTemplate(
                        input_variables=["question", "prediction", "answer"],
                        template = JUDGE_PROMPT,
                        )
