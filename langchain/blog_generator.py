import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pprint import pprint

##based on complete langchain 

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


# Initialize the ChatGroq LLM with the specified model
lm = ChatGroq(model="gemma2-9b-it")

# Agent 1: Title Generation
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="You are a creative writer. Given the topic '{topic}', generate an engaging title for an article."
)
title_chain = LLMChain(llm=lm, prompt=title_prompt, output_key="title")

# Agent 2: Content Generation (Article Outline or Draft Content)
content_prompt = PromptTemplate(
    input_variables=["title"],
    template="You are a subject matter expert. Based on the title '{title}', generate a detailed article outline or draft content that covers the topic comprehensively."
)
content_chain = LLMChain(llm=lm, prompt=content_prompt, output_key="content")

# Agent 3: SEO Optimization
seo_prompt = PromptTemplate(
    input_variables=["content"],
    template="You are an SEO specialist. Analyze the following article content and provide SEO recommendations. Include a meta title, meta description, and a list of target keywords:\n\n{content}\n\nSEO Recommendations:"
)
seo_chain = LLMChain(llm=lm, prompt=seo_prompt, output_key="seo")

def run_pipeline(user_topic):
    # Agent 1: Generate the Title
    title = title_chain.run(topic=user_topic)
    print("\nGenerated Title:")
    print(title)
    
    # Agent 2: Generate Content Based on the Title
    content = content_chain.run(title=title)
    print("\nGenerated Content/Outline:")
    print(content)
    
    # Agent 3: Generate SEO Recommendations Based on the Content
    seo_recommendations = seo_chain.run(content=content)
    print("\nSEO Recommendations:")
    print(seo_recommendations)
    
    return {
        "title": title,
        "content": content,
        "seo": seo_recommendations
    }

if __name__ == "__main__":
    user_topic = input("Enter your topic: ")
    output = run_pipeline(user_topic)
    print(output)
