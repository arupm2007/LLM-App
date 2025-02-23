import os
import json
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Load environment variables and set API key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define a state with an input topic and output fields.
class State(TypedDict):
    topic: str
    title: str
    blog: str
    seo: str

# Initialize the ChatGroq LLM with the specified model.
llm = ChatGroq(model="gemma2-9b-it")

def title_generator(state: State):
    msg = llm.invoke(f"Write best possible single title for the topic : {state['topic']}")
    return {"title": msg.content}

def blog_generator(state: State):
    msg = llm.invoke(f"Write detailed blog for the title : {state['title']}")
    return {"blog": msg.content}

def seo_generator(state: State) -> State:
    msg = llm.invoke(f"Write SEO contents for the blog: {state['blog']}")
    return {"seo": msg.content}

# A simple function to clean markdown formatting from text.
def clean_text(text: str) -> str:
    # Replace newlines with a space
    cleaned = text.replace("\n", " ")
    cleaned = re.sub(r"##", "", cleaned)
    # Remove markdown asterisks and extra spaces
    cleaned = re.sub(r"\*\*", "", cleaned)
    cleaned = re.sub(r"\*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# Build the graph using StateGraph with START and END.
builder = StateGraph(State)

# Add nodes with their corresponding functions.
builder.add_node("title_generator", title_generator)
builder.add_node("blog_generator", blog_generator)
builder.add_node("seo_generator", seo_generator)

# Define the logic edges.
builder.add_edge(START, "title_generator")
builder.add_edge("title_generator", "blog_generator")
builder.add_edge("blog_generator", "seo_generator")
builder.add_edge("seo_generator", END)

# Compile the graph.
graph = builder.compile()

# Get the directory of the current program/script.
program_directory = os.path.dirname(os.path.realpath(__file__))

# Get the Mermaid diagram as PNG binary data.
mermaid_png = graph.get_graph().draw_mermaid_png()

# Save the Mermaid diagram to a file in the program directory.
png_path = os.path.join(program_directory, "workflow.png")
with open(png_path, "wb") as img_file:
    img_file.write(mermaid_png)

# Optionally, display the workflow image (if using an IPython environment)
display(Image(mermaid_png))

# Invoke the graph with an initial state.
state = graph.invoke({"topic": "Cat sitting on a wall"})

# Clean the output for each string field.
cleaned_state = {}
for key, value in state.items():
    if isinstance(value, str):
        cleaned_state[key] = clean_text(value)
    else:
        cleaned_state[key] = value

# Convert the cleaned state to JSON and save it to a file.
json_output = json.dumps(cleaned_state, indent=2)
json_path = os.path.join(program_directory, "output.json")
with open(json_path, "w") as json_file:
    json_file.write(json_output)

# Also print the cleaned JSON output to the console.
print(json_output)
