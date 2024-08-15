import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer
from langchain_core.utils.function_calling import convert_to_openai_function

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

parser = JsonOutputToolsParser(return_id=True)

AnswerSchema = convert_to_openai_function(AnswerQuestion, strict=True)

responder_llm = llm.with_structured_output(
    AnswerSchema,
    method="json_schema"
)

actor_prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """You are an expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format."
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 words per answer"
)

first_responder = first_responder_prompt_template | responder_llm

revise_instructions = """Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more
than 250 words.
"""

revisor_llm = llm.with_structured_output(
    ReviseAnswer,
    method="json_schema"
)

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | revisor_llm

if __name__ == "__main__":
    human_message = ["Write about e-sports analytics problem domain, list startups that do that and raised capital."]

    res = first_responder.invoke(input={"messages": human_message})

    print(res)
