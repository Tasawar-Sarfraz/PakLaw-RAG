from groq import Groq


class GovernmentLLM:

    def __init__(
        self,
        api_key,
        model="openai/gpt-oss-120b"
    ):

        self.client = Groq(
            api_key=api_key
        )

        self.model = model


    def generate(
        self,
        question,
        context
    ):

        system_prompt = """
You are a Government Information Assistant.

Your job is to answer questions using ONLY
the provided government document context.

STRICT RULES:

1. Do not invent information.

2. Do not use outside knowledge as government policy.

3. If the answer is not present in the provided
   context, clearly say:

   "The provided documents do not contain enough
   information to answer this question."

4. Preserve the meaning of tables and columns.

5. If information comes from a table, present it
   in a table when appropriate.

6. Mention the relevant government department.

7. Mention the document/source and page when
   useful.

8. If multiple documents provide information,
   clearly distinguish them.

9. Do not merge unrelated information from
   different departments.

10. Give a clear and structured answer.

11. Do not claim something is present in the
    documents unless it is actually present
    in the provided context.
"""


        user_prompt = f"""
QUESTION:

{question}


DOCUMENT CONTEXT:

{context}
"""


        response = (
            self.client
            .chat
            .completions
            .create(

                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=0.1
            )
        )


        return (
            response
            .choices[0]
            .message
            .content
        )