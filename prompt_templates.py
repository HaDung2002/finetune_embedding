QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
Generate {num_questions_per_chunk} questions written in English and \
{num_questions_per_chunk} questions written in Korean. \
Restrict the questions to the context information provided.
"""